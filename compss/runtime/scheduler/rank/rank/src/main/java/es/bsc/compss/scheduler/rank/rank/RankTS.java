/*
 *  Copyright 2002-2025 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package es.bsc.compss.scheduler.rank.rank;

import es.bsc.compss.components.impl.ResourceScheduler;
import es.bsc.compss.scheduler.exceptions.BlockedActionException;
import es.bsc.compss.scheduler.exceptions.UnassignedActionException;
import es.bsc.compss.scheduler.rank.base.RankBaseTS;
import es.bsc.compss.scheduler.rank.base.RankSchedulingOptimizer;
import es.bsc.compss.scheduler.types.ActionOrchestrator;
import es.bsc.compss.scheduler.types.AllocatableAction;
import es.bsc.compss.scheduler.types.ObjectValue;
import es.bsc.compss.scheduler.types.Profile;
import es.bsc.compss.scheduler.types.Score;
import es.bsc.compss.types.allocatableactions.ExecutionAction;
import es.bsc.compss.types.parameter.impl.BasicTypeParameter;
import es.bsc.compss.types.parameter.impl.CollectiveParameter;
import es.bsc.compss.types.parameter.impl.FileParameter;
import es.bsc.compss.types.parameter.impl.Parameter;
import es.bsc.compss.types.resources.WorkerResourceDescription;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;


public class RankTS extends RankBaseTS {

    @Override
    protected String getLoggerPrefix() {
        return "[RankTS]";
    }

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    /**
     * Constructs a new RankTS instance.
     *
     * @param orchestrator Element that orders the execution of actions.
     */
    public RankTS(ActionOrchestrator orchestrator) {
        super(orchestrator);
    }

    // -------------------------------------------------------------------------
    // Scheduling operations
    // -------------------------------------------------------------------------

    @Override
    protected <T extends WorkerResourceDescription> void handleDependencyFreeActions(
        List<AllocatableAction> dataFreeActions, List<AllocatableAction> resourceFreeActions,
        List<AllocatableAction> blockedCandidates, ResourceScheduler<T> resource) {

        boolean addedAny = false;

        manageUpgradedActions(resource);

        PriorityQueue<ObjectValue<AllocatableAction>> executableActions = new PriorityQueue<>();

        // Process newly data-free actions.
        for (AllocatableAction freeAction : dataFreeActions) {

            if (freeAction instanceof ExecutionAction) {
                ExecutionAction execFreeAction = (ExecutionAction) freeAction;
                int taskRank = execFreeAction.getTask().getRank();

                if (taskRank != 0) {
                    boolean hasLowerRankedTask = false;

                    // Check among the current batch of newly data-free actions.
                    for (AllocatableAction other : dataFreeActions) {
                        if (other == freeAction || !(other instanceof ExecutionAction)) {
                            continue;
                        }
                        int otherRank = ((ExecutionAction) other).getTask().getRank();
                        if (otherRank >= taskRank) {
                            continue; // not a lower-ranked candidate
                        } else {
                            hasLowerRankedTask = true;
                            break;
                        }
                    }
                    // If not found yet, also check already-deferred actions.
                    if (!hasLowerRankedTask) {
                        for (AllocatableAction deferred : deferredActions.keySet()) {
                            if (!(deferred instanceof ExecutionAction)) {
                                continue;
                            }
                            int deferredRank = ((ExecutionAction) deferred).getTask().getRank();
                            if (deferredRank >= taskRank) {
                                continue;
                            } else {
                                hasLowerRankedTask = true;
                                break;
                            }
                        }
                    }
                    if (hasLowerRankedTask) {
                        // A task with higher scheduling priority exists: defer.
                        deferredActions.put(freeAction, System.currentTimeMillis());
                        addedAny = true;
                        LOGGER.debug(
                            "[RankTS] Task rank " + taskRank + " deferred — awaiting lower-ranked task: " + freeAction);
                    } else {
                        // This action has the lowest rank: schedule now.
                        Score actionScore = generateActionScore(freeAction);
                        executableActions.add(new ObjectValue<>(freeAction, actionScore));
                        LOGGER.debug("[RankTS] Task rank " + taskRank
                            + " is lowest available — scheduling immediately: " + freeAction);
                    }

                } else {
                    // rank == 0 means highest priority: schedule immediately.
                    Score actionScore = generateActionScore(freeAction);
                    executableActions.add(new ObjectValue<>(freeAction, actionScore));
                }
            } else {
                // Non-ExecutionAction (system action): schedule immediately, no rank logic.
                Score actionScore = generateActionScore(freeAction);
                executableActions.add(new ObjectValue<>(freeAction, actionScore));
            }
        }

        if (addedAny) {
            deferredDirty = true;
        }

        boolean canExecute = true;
        boolean readyQueueEmpty = readyQueue.isEmpty();
        boolean executableActionsEmpty = executableActions.isEmpty();

        while (canExecute && (!executableActionsEmpty || !readyQueueEmpty)) {
            ObjectValue<AllocatableAction> topReadyQueue = readyQueue.peek();
            ObjectValue<AllocatableAction> topExecutableActions = executableActions.peek();

            Score topReadyQueueScore = readyQueueEmpty ? null : topReadyQueue.getScore();
            Score topExecutableActionsScore = executableActionsEmpty ? null : topExecutableActions.getScore();

            ObjectValue<AllocatableAction> topPriority =
                Score.isBetter(topReadyQueueScore, topExecutableActionsScore) ? topReadyQueue : topExecutableActions;

            AllocatableAction aa = topPriority.getObject();
            try {
                scheduleAndLaunchAction(aa, topPriority.getScore());

                if (topPriority == topReadyQueue) {
                    readyQueue.poll();
                    addedActions.remove(aa);
                    readyQueueEmpty = readyQueue.isEmpty();
                } else {
                    executableActions.poll();
                    executableActionsEmpty = executableActions.isEmpty();
                }
            } catch (UnassignedActionException uae) {
                canExecute = false;
            } catch (BlockedActionException bae) {
                addToBlocked(aa);
            }
        }

        // Merge remaining executable actions back into the ready queue.
        if (!executableActions.isEmpty()) {
            readyQueue.addAll(executableActions);
        }
    }
}
