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
package es.bsc.compss.scheduler.rank.prediction;

import es.bsc.compss.components.impl.ResourceScheduler;
import es.bsc.compss.scheduler.exceptions.BlockedActionException;
import es.bsc.compss.scheduler.exceptions.UnassignedActionException;
import es.bsc.compss.scheduler.rank.base.RankBaseTS;
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


/**
 * Task scheduler that extends {@link RankBaseTS} combining a similarity-based look-ahead mechanism with a rank-based
 * fallback deferral policy.
 */
public class PredictionTS extends RankBaseTS {

    /**
     * Half-width of the DAG-depth window used to filter comparison candidates. A task {@code taskOld} is eligible for
     * comparison only when {@code |depth(taskOld) - depth(newTask)| <= DEPTH_WINDOW}.
     */
    private static final int DEPTH_WINDOW = 3;

    /**
     * Upper bound on the number of tasks retained in the in-memory sliding window.
     */
    private static final int MAX_RECENT_WINDOW_SIZE = 1_000;

    /**
     * Minimum similarity score required for hint generation. Tasks whose score falls below this threshold are silently
     * ignored.
     */
    private static final double SIMILARITY_THRESHOLD = 0.7;

    // -------------------------------------------------------------------------
    // Similarity look-ahead fields
    // -------------------------------------------------------------------------

    /**
     * Sliding window of recently seen execution tasks, bounded by {@link #MAX_RECENT_WINDOW_SIZE}. Entries are evicted
     * FIFO when the bound is reached.
     */
    private final Deque<AllocatableAction> recentTaskWindow = new ArrayDeque<>();

    /**
     * Reverse map from task ID to the corresponding {@link AllocatableAction}, used when processing pending hint
     * requests after a previously unexecuted successor completes.
     */
    private final Map<Long, AllocatableAction> taskIdToAction = new HashMap<>();

    /** Shared task-graph cache providing DAG structure and per-node features. */
    private final TaskGraphCache taskGraphCache;

    /**
     * Stateful engine implementing incremental feature-space tracking and similarity maths.
     */
    private final SimilarityEngine similarityEngine;

    /**
     * Pluggable similarity function; defaults to {@link WLSimilarityFunction}.
     */
    private final SimilarityFunction similarityFunction;

    /**
     * Maps each {@code newTask} to the list of {@link SuccessorHint} objects generated from the matching
     * {@code oldTask}'s successors. Entries are consumed in {@link #handleDependencyFreeActions} when a predicted
     * successor becomes dependency-free, and removed once all hints in the list have been consumed.
     */
    private final Map<AllocatableAction, List<SuccessorHint>> successorHintMap = new HashMap<>();


    @Override
    protected String getLoggerPrefix() {
        return "[PredictionTS]";
    }

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    /**
     * Constructs a new PredictionTS instance.
     * 
     * @param orchestrator Element that orders the execution of actions.
     */
    public PredictionTS(ActionOrchestrator orchestrator) {
        super(orchestrator);
        this.taskGraphCache = new TaskGraphCache();
        this.similarityEngine = new SimilarityEngine(taskGraphCache);
        this.similarityFunction =
            new WLSimilarityFunction(taskGraphCache, similarityEngine, /* hops */ 2, /* r */ 0.8, /* alpha */ 0.0);
    }

    // -------------------------------------------------------------------------
    // Entry point: new task arrival
    // -------------------------------------------------------------------------

    /**
     * Registers the action in the {@link TaskGraphCache}, updates the {@link SimilarityEngine} feature space, evaluates
     * similarity against depth-window candidates, and builds hints if a match is found before delegating to the base
     * scheduling logic. Non-execution actions bypass steps 1–4 and are forwarded directly to the base implementation.
     *
     * @param action The action entering the scheduler.
     * @param actionScore The scheduling score computed for the action.
     * @throws BlockedActionException If no compatible worker is available.
     */
    @Override
    public void scheduleAction(AllocatableAction action, Score actionScore) throws BlockedActionException {
        if (action instanceof ExecutionAction) {

            ExecutionAction execAction = (ExecutionAction) action;
            int coreId = execAction.getCoreId();

            Map<String, String> categorical = new HashMap<>();
            Map<String, double[]> numerical = new HashMap<>();

            categorical.put("task_name", execAction.getTask().getTaskDescription().getName());
            List<? extends Parameter> parameters = execAction.getTask().getParameters();
            int counter = 1;

            for (Parameter p : parameters) {
                categorical.put("param" + counter, p.getName());
                counter++;

                if (!p.isPotentialDependency()) {
                    if (p instanceof BasicTypeParameter) {
                        BasicTypeParameter sp = (BasicTypeParameter) p;
                        if (sp.getValue() instanceof Number) {
                            double[] spNumArray = new double[] { ((Number) sp.getValue()).doubleValue() };
                            numerical.put(sp.getName(), spNumArray);
                        }
                    } else if (p instanceof CollectiveParameter) {
                        double[] spNumArray = new double[] { (double) ((CollectiveParameter) p).getElements().size() };
                        numerical.put(p.getName(), spNumArray);
                    } else if (p instanceof FileParameter) {
                        ; // TODO: retrieve file size
                    }
                }
            }

            TaskFeatures features = new TaskFeatures(coreId, categorical, numerical);

            List<Long> predecessorIds = new ArrayList<>();
            for (AllocatableAction pred : execAction.getDataPredecessors()) {
                predecessorIds.add(pred.getId());
            }

            taskGraphCache.registerTask(execAction.getId(), features, predecessorIds);
            taskIdToAction.put(execAction.getId(), execAction);
            similarityEngine.updateFeatureSpace(execAction.getId());
            evaluateSimilarityAndBuildHints(execAction);

            if (recentTaskWindow.size() >= MAX_RECENT_WINDOW_SIZE) {
                AllocatableAction evicted = recentTaskWindow.pollFirst();
                successorHintMap.remove(evicted);
            }
            recentTaskWindow.addLast(execAction);
        }

        // Delegate base scheduling logic (ready queue / immediate dispatch).
        super.scheduleAction(action, actionScore);
    }

    // -------------------------------------------------------------------------
    // Action completion
    // -------------------------------------------------------------------------

    /**
     * Updates the {@link TaskGraphCache} with the resource and execution profile of the completed action, then resolves
     * any pending hint requests registered while the action was still unexecuted. For each pending request a
     * {@link SuccessorHint} (rank 0) is created and stored in {@code successorHintMap}, enabling the hint-based
     * scheduling logic in {@link #handleDependencyFreeActions} to use it in future cycles.
     *
     * @param action The action that has just completed.
     */
    @Override
    public void actionCompleted(AllocatableAction action) {
        if (action.getCoreId() != null) {
            ResourceScheduler<?> rs = action.getAssignedResource();
            Profile profile = action.getProfile();

            Map<Long, Double> pending = taskGraphCache.updateOnCompletion(action.getId(), rs, profile);

            for (Map.Entry<Long, Double> entry : pending.entrySet()) {
                long newTaskId = entry.getKey();
                double confidence = entry.getValue();
                AllocatableAction newTaskAction = taskIdToAction.get(newTaskId);
                TaskGraphCache.NodeInfo succInfo = taskGraphCache.getNode(action.getId());

                if (newTaskAction != null && succInfo != null) {
                    SuccessorHint hint =
                        new SuccessorHint(succInfo.features.getCoreId(), succInfo.executionProfile, confidence, 0);
                    successorHintMap.computeIfAbsent(newTaskAction, k -> new ArrayList<>()).add(hint);
                }
            }
        }
        super.actionCompleted(action);
    }

    // -------------------------------------------------------------------------
    // Similarity evaluation and hint generation
    // -------------------------------------------------------------------------

    /**
     * For each task in the sliding window that passes the depth-window and sibling filters, computes a similarity score
     * against {@code taskNew}. When the score exceeds {@link #SIMILARITY_THRESHOLD}, retrieves the direct successors of
     * the matched task: successors with a known execution profile are sorted by descending average execution time
     * (Longest Job First) and stored as ranked {@link SuccessorHint}s in {@code successorHintMap[taskNew]}; successors
     * that have not yet completed register a pending-hint request to be fulfilled in {@link #actionCompleted}.
     *
     * @param taskNew Newly arrived execution action to evaluate.
     */
    private void evaluateSimilarityAndBuildHints(AllocatableAction taskNew) {
        TaskGraphCache.NodeInfo newInfo = taskGraphCache.getNode(taskNew.getId());
        if (newInfo == null) {
            return;
        }
        int depthNew = newInfo.depth;
        Set<Long> newPredSet = new HashSet<>(newInfo.predecessorIds);

        for (AllocatableAction taskOld : recentTaskWindow) {

            if (newInfo.predecessorIds.contains(taskOld.getId())) {
                continue;
            }

            TaskGraphCache.NodeInfo oldInfo = taskGraphCache.getNode(taskOld.getId());
            if (oldInfo == null) {
                continue;
            }
            if (Math.abs(oldInfo.depth - depthNew) > DEPTH_WINDOW) {
                continue;
            }

            Set<Long> oldPredSet = new HashSet<>(oldInfo.predecessorIds);
            if (!newPredSet.isEmpty() && newPredSet.equals(oldPredSet)) {
                continue;
            }

            double sim = similarityFunction.compute(taskNew.getId(), taskOld.getId());

            if (sim > SIMILARITY_THRESHOLD) {
                LOGGER.debug(
                    "[PredictionTS] Similarity match: taskNew=" + taskNew + " taskOld=" + taskOld + " score=" + sim);

                List<TaskGraphCache.NodeInfo> successors = taskGraphCache.getSuccessors(taskOld.getId());

                List<TaskGraphCache.NodeInfo> completedSuccessors = new ArrayList<>();
                for (TaskGraphCache.NodeInfo succ : successors) {
                    if (succ.executionProfile != null) {
                        completedSuccessors.add(succ);
                    } else {
                        succ.pendingHintRequests.put(taskNew.getId(), sim);
                    }
                }

                Comparator<TaskGraphCache.NodeInfo> comparator =
                    Comparator.comparingLong(s -> s.executionProfile.getExecutionTime());
                completedSuccessors.sort(comparator.reversed());

                for (int rankIdx = 0; rankIdx < completedSuccessors.size(); rankIdx++) {
                    TaskGraphCache.NodeInfo succ = completedSuccessors.get(rankIdx);
                    SuccessorHint hint =
                        new SuccessorHint(succ.features.getCoreId(), succ.executionProfile, sim, rankIdx);
                    successorHintMap.computeIfAbsent(taskNew, k -> new ArrayList<>()).add(hint);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Scheduling operations
    // -------------------------------------------------------------------------

    /**
     * Handles actions that have just become free of data dependencies applying a two-tier policy:
     * <ol>
     * <li><b>Hint-based (siblings only).</b> If a {@link SuccessorHint} is found for the action among its predecessors'
     * hint maps, hint ranks are used to order sibling execution: the action with the lowest hint rank is submitted
     * immediately; all others are deferred into {@code deferredActions} until released by the optimizer.</li>
     * <li><b>Rank-based fallback (siblings only).</b> If no hint is available and the action is an
     * {@link ExecutionAction} with rank {@literal >} 0, the task rank is used instead: the action is deferred when any
     * sibling (task sharing at least one common predecessor) with a lower rank exists in {@code dataFreeActions} or
     * {@code deferredActions}; otherwise it is submitted immediately. Rank-0 actions are always submitted
     * immediately.</li>
     * </ol>
     * Resource-free actions ({@code resourceFreeActions}) are not handled by this scheduler variant. Deferred actions
     * are eventually released in ascending rank order by
     * {@link es.bsc.compss.scheduler.rank.base.RankSchedulingOptimizer}.
     */
    @Override
    protected <T extends WorkerResourceDescription> void handleDependencyFreeActions(
        List<AllocatableAction> dataFreeActions, List<AllocatableAction> resourceFreeActions,
        List<AllocatableAction> blockedCandidates, ResourceScheduler<T> resource) {

        boolean addedAny = false;

        manageUpgradedActions(resource);

        PriorityQueue<ObjectValue<AllocatableAction>> executableActions = new PriorityQueue<>();

        for (AllocatableAction freeAction : dataFreeActions) {

            SuccessorHint matchingHint = null;
            AllocatableAction hintPredecessor = null;

            for (TaskGraphCache.NodeInfo predTask : taskGraphCache.getPredecessors(freeAction.getId())) {
                AllocatableAction predecessor = taskIdToAction.get(predTask.taskId);
                List<SuccessorHint> hints = successorHintMap.get(predecessor);
                if (hints == null) {
                    continue;
                }
                for (SuccessorHint hint : hints) {
                    if (!hint.isConsumed() && hint.getCoreId() == freeAction.getCoreId()) {
                        matchingHint = hint;
                        hintPredecessor = predecessor;
                        break;
                    }
                }
                if (matchingHint != null) {
                    break;
                }
            }

            if (matchingHint != null) {
                List<SuccessorHint> siblingHints = successorHintMap.get(hintPredecessor);
                int freeActionRank = matchingHint.getRank();
                boolean hasLowerRankedSibling = false;
                for (SuccessorHint sibling : siblingHints) {
                    if (!sibling.isConsumed() && sibling != matchingHint && sibling.getRank() < freeActionRank) {
                        hasLowerRankedSibling = true;
                        break;
                    }
                }

                if (!hasLowerRankedSibling) {
                    matchingHint.markConsumed();
                    Score actionScore = generateActionScore(freeAction);
                    executableActions.add(new ObjectValue<>(freeAction, actionScore));
                    LOGGER.debug("[PredictionTS] Hint rank " + freeActionRank + " is lowest — scheduling immediately: "
                        + freeAction);
                } else {
                    deferredActions.put(freeAction, System.currentTimeMillis());
                    addedAny = true;
                    LOGGER.debug("[PredictionTS] Hint rank " + freeActionRank
                        + " deferred — awaiting lower-ranked sibling: " + freeAction);
                }

                boolean allConsumed = true;
                for (SuccessorHint sibling : siblingHints) {
                    if (!sibling.isConsumed()) {
                        allConsumed = false;
                        break;
                    }
                }
                if (allConsumed) {
                    successorHintMap.remove(hintPredecessor);
                }

            } else {
                // Rank-based fallback: sibling check via shared predecessors.
                if (freeAction instanceof ExecutionAction) {
                    ExecutionAction execFreeAction = (ExecutionAction) freeAction;
                    int taskRank = execFreeAction.getTask().getRank();

                    if (taskRank != 0) {
                        Set<Long> freeActionPredIds = getPredecessorIds(freeAction);
                        boolean hasLowerRankedSibling = false;

                        for (AllocatableAction other : dataFreeActions) {
                            if (other == freeAction || !(other instanceof ExecutionAction)) {
                                continue;
                            }
                            int otherRank = ((ExecutionAction) other).getTask().getRank();
                            if (otherRank >= taskRank) {
                                continue;
                            }
                            Set<Long> otherPredIds = getPredecessorIds(other);
                            if (!java.util.Collections.disjoint(freeActionPredIds, otherPredIds)) {
                                hasLowerRankedSibling = true;
                                break;
                            }
                        }

                        if (!hasLowerRankedSibling) {
                            for (AllocatableAction deferred : deferredActions.keySet()) {
                                if (!(deferred instanceof ExecutionAction)) {
                                    continue;
                                }
                                int deferredRank = ((ExecutionAction) deferred).getTask().getRank();
                                if (deferredRank >= taskRank) {
                                    continue;
                                }
                                Set<Long> deferredPredIds = getPredecessorIds(deferred);
                                if (!java.util.Collections.disjoint(freeActionPredIds, deferredPredIds)) {
                                    hasLowerRankedSibling = true;
                                    break;
                                }
                            }
                        }

                        if (hasLowerRankedSibling) {
                            deferredActions.put(freeAction, System.currentTimeMillis());
                            addedAny = true;
                            LOGGER.debug("[PredictionTS] Task rank " + taskRank
                                + " deferred (rank-based) — awaiting lower-ranked sibling: " + freeAction);
                        } else {
                            Score actionScore = generateActionScore(freeAction);
                            executableActions.add(new ObjectValue<>(freeAction, actionScore));
                            LOGGER.debug("[PredictionTS] Task rank " + taskRank
                                + " is lowest available (rank-based) — scheduling immediately: " + freeAction);
                        }

                    } else {
                        Score actionScore = generateActionScore(freeAction);
                        executableActions.add(new ObjectValue<>(freeAction, actionScore));
                    }
                } else {
                    Score actionScore = generateActionScore(freeAction);
                    executableActions.add(new ObjectValue<>(freeAction, actionScore));
                }
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

        if (!executableActions.isEmpty()) {
            readyQueue.addAll(executableActions);
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Returns the set of predecessor task IDs for the given action as recorded in the {@link TaskGraphCache}. Returns
     * an empty set if the action is not registered or has no predecessors.
     *
     * @param action The action whose predecessors are queried.
     * @return A set of task IDs corresponding to the action's data predecessors.
     */
    private Set<Long> getPredecessorIds(AllocatableAction action) {
        Set<Long> predIds = new HashSet<>();
        for (TaskGraphCache.NodeInfo pred : taskGraphCache.getPredecessors(action.getId())) {
            predIds.add(pred.taskId);
        }
        return predIds;
    }
}