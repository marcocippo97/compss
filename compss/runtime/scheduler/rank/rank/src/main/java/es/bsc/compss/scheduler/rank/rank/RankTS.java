package es.bsc.compss.scheduler.rank.rank;

import es.bsc.compss.components.impl.ResourceScheduler;
import es.bsc.compss.scheduler.exceptions.BlockedActionException;
import es.bsc.compss.scheduler.exceptions.UnassignedActionException;
import es.bsc.compss.scheduler.rank.base.RankBaseTS;
import es.bsc.compss.scheduler.types.ActionOrchestrator;
import es.bsc.compss.scheduler.types.AllocatableAction;
import es.bsc.compss.scheduler.types.ObjectValue;
import es.bsc.compss.scheduler.types.Score;
import es.bsc.compss.types.allocatableactions.ExecutionAction;
import es.bsc.compss.types.resources.WorkerResourceDescription;

import java.util.List;
import java.util.PriorityQueue;


/**
 * Task scheduler that extends {@link RankBaseTS} with a pure rank-based scheduling policy. When a group of tasks
 * becomes data-free, each task's rank is embedded into its {@link Score} priority via {@link #generateRankedScore}:
 * rank 0 (highest scheduling priority) receives the largest priority value and is therefore dispatched first by the
 * ready queue, while higher rank values are dispatched later in ascending order.
 */
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

    /**
     * Handles actions that have just become free of data dependencies by embedding each action's rank into its
     * scheduling score and submitting all actions to the executable queue. The underlying {@link PriorityQueue}
     * ordering ensures that rank-0 tasks are always dispatched before higher-ranked ones when resources are contested.
     * For {@link ExecutionAction}s the effective rank is read from the task; all other action types are treated as rank
     * 0 and submitted immediately.
     */
    @Override
    protected <T extends WorkerResourceDescription> void handleDependencyFreeActions(
        List<AllocatableAction> dataFreeActions, List<AllocatableAction> resourceFreeActions,
        List<AllocatableAction> blockedCandidates, ResourceScheduler<T> resource) {

        manageUpgradedActions(resource);

        PriorityQueue<ObjectValue<AllocatableAction>> executableActions = new PriorityQueue<>();

        for (AllocatableAction freeAction : dataFreeActions) {
            Score actionScore = generateActionScore(freeAction);
            executableActions.add(new ObjectValue<>(freeAction, actionScore));
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
}