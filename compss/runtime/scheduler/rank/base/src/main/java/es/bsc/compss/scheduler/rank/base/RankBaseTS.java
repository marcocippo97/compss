package es.bsc.compss.scheduler.rank.base;

import es.bsc.compss.components.impl.ResourceScheduler;
import es.bsc.compss.components.impl.TaskScheduler;
import es.bsc.compss.scheduler.exceptions.BlockedActionException;
import es.bsc.compss.scheduler.exceptions.UnassignedActionException;
import es.bsc.compss.scheduler.types.ActionOrchestrator;
import es.bsc.compss.scheduler.types.AllocatableAction;
import es.bsc.compss.scheduler.types.ObjectValue;
import es.bsc.compss.scheduler.types.Score;
import es.bsc.compss.types.allocatableactions.ExecutionAction;
import es.bsc.compss.types.resources.Worker;
import es.bsc.compss.types.resources.WorkerResourceDescription;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import org.json.JSONObject;


/**
 * Abstract base task scheduler that provides rank-aware scheduling infrastructure. Rank is embedded directly into the
 * {@link Score#priority} field so that the {@link PriorityQueue} ordering replaces any explicit deferral mechanism:
 * tasks with rank 0 (highest scheduling priority) are always dispatched before tasks with a higher rank value. Concrete
 * subclasses implement {@link #handleDependencyFreeActions} to define how the effective rank is determined (pure
 * task-rank or hint-assisted).
 */
public abstract class RankBaseTS extends TaskScheduler {

    /** Priority queue of ready actions sorted by scheduling score. */
    protected final PriorityQueue<ObjectValue<AllocatableAction>> readyQueue;

    /** Actions that have been upgraded and must be handled with priority. */
    protected Set<AllocatableAction> upgradedActions;

    /** Map from action to its current ObjectValue wrapper in the ready queue. */
    protected final Map<AllocatableAction, ObjectValue<AllocatableAction>> addedActions;


    /**
     * Returns the log-line prefix used by this scheduler variant (e.g. {@code "[RankTS]"}). Implemented by concrete
     * subclasses to distinguish log output.
     */
    protected abstract String getLoggerPrefix();

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    /**
     * Constructs a new RankBaseTS instance.
     *
     * @param orchestrator Element that orders the execution of actions.
     */
    public RankBaseTS(ActionOrchestrator orchestrator) {
        super(orchestrator);
        this.readyQueue = new PriorityQueue<>();
        this.upgradedActions = new HashSet<>();
        this.addedActions = new HashMap<>();
        LOGGER.debug(getLoggerPrefix() + " Initialising task scheduler");
    }

    // -------------------------------------------------------------------------
    // Action lifecycle overrides
    // -------------------------------------------------------------------------

    /**
     * Moves an action to the set of upgraded actions and removes it from the ready queue so that it can be re-scheduled
     * with elevated priority.
     *
     * @param action The action to upgrade.
     */
    @Override
    public void upgradeAction(AllocatableAction action) {
        if (DEBUG) {
            LOGGER.debug(getLoggerPrefix() + " Upgrading action " + action);
        }
        upgradedActions.add(action);
        ObjectValue<AllocatableAction> obj = addedActions.remove(action);
        readyQueue.remove(obj);
    }

    /**
     * Schedules a newly arrived action. Actions without data predecessors are dispatched immediately or added to the
     * ready queue; actions with predecessors are held until their dependencies are resolved. Subclasses may override
     * this method to register additional per-action metadata before delegating to this base logic.
     *
     * @param action The action entering the scheduler.
     * @param actionScore The scheduling score computed for the action.
     * @throws BlockedActionException If no compatible worker is available.
     */
    @Override
    public void scheduleAction(AllocatableAction action, Score actionScore) throws BlockedActionException {
        if (!action.hasDataPredecessors()) {
            if (upgradedActions.isEmpty()) {
                ObjectValue<AllocatableAction> topReady = readyQueue.peek();
                if (topReady == null || actionScore.isBetter(topReady.getScore())) {
                    try {
                        action.schedule(actionScore);
                    } catch (UnassignedActionException uae) {
                        addActionToReadyQueue(action, actionScore);
                    }
                } else {
                    if (action.getCompatibleWorkers().isEmpty()) {
                        throw new BlockedActionException();
                    }
                    addActionToReadyQueue(action, actionScore);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Protected helpers (available to subclasses)
    // -------------------------------------------------------------------------

    /**
     * Builds a priority queue by scoring each action in {@code actions} with {@link #generateActionScore}, used to
     * order upgraded actions before dispatch.
     *
     * @param actions The actions to score and sort.
     * @return A {@link PriorityQueue} of wrapped actions ordered by score.
     */
    protected PriorityQueue<ObjectValue<AllocatableAction>> sortActions(Iterable<AllocatableAction> actions) {
        if (DEBUG) {
            LOGGER.debug(getLoggerPrefix() + " Sorting upgraded actions.");
        }
        PriorityQueue<ObjectValue<AllocatableAction>> sorted = new PriorityQueue<>();
        for (AllocatableAction action : actions) {
            Score score = generateActionScore(action);
            sorted.add(new ObjectValue<>(action, score));
        }
        return sorted;
    }

    /**
     * Attempts to schedule all pending upgraded actions on the given resource in score order, removing each
     * successfully scheduled action from {@code upgradedActions}.
     *
     * @param resource The resource on which to try scheduling upgraded actions.
     */
    protected void manageUpgradedActions(ResourceScheduler<?> resource) {
        if (upgradedActions.isEmpty()) {
            return;
        }
        if (DEBUG) {
            LOGGER.debug(getLoggerPrefix() + " Managing " + upgradedActions.size() + " upgraded actions.");
        }
        PriorityQueue<ObjectValue<AllocatableAction>> executableActions = sortActions(upgradedActions);
        while (!executableActions.isEmpty()) {
            ObjectValue<AllocatableAction> obj = executableActions.poll();
            AllocatableAction freeAction = obj.getObject();
            if (freeAction.getCompatibleWorkers().contains(resource) && resource.canRunSomething()) {
                try {
                    freeAction.schedule(resource, obj.getScore());
                    tryToLaunch(freeAction);
                    upgradedActions.remove(freeAction);
                } catch (UnassignedActionException e) {
                    // Action can be scheduled on a different resource; continue.
                }
            }
        }
    }

    /**
     * Wraps the action in an {@link ObjectValue} and inserts it into both {@code readyQueue} and {@code addedActions}.
     *
     * @param action The action to enqueue.
     * @param actionScore The score to associate with the action.
     */
    protected void addActionToReadyQueue(AllocatableAction action, Score actionScore) {
        ObjectValue<AllocatableAction> obj = new ObjectValue<>(action, actionScore);
        addedActions.put(action, obj);
        readyQueue.add(obj);
    }

    /**
     * Returns a new {@link Score} derived from the base score of {@code action} but with {@code priority} set to
     * {@code Integer.MAX_VALUE - rank}.
     *
     * @param action The action for which to compute the score.
     * @return A new {@link Score} with rank-derived priority.
     */
    public Score generateActionScore(AllocatableAction action) {
        int rank = 0;
        if (action instanceof ExecutionAction) {
            rank = ((ExecutionAction) action).getTask().getRank();
            LOGGER.debug(getLoggerPrefix() + " Generating Score with rank " + rank + " from " + action);
        }
        long priority = (long) Integer.MAX_VALUE - rank;
        return new Score(priority, action.getGroupPriority(), 0, 0, 0);
    }

    public Score generateActionScore(AllocatableAction action, int rank) {
        long priority = (long) Integer.MAX_VALUE - rank;
        return new Score(priority, action.getGroupPriority(), 0, 0, 0);
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Assigns the action to a resource and immediately attempts to launch it.
     *
     * @param aa The action to schedule and launch.
     * @param score The score to use for resource selection.
     * @throws BlockedActionException If no suitable resource is available.
     * @throws UnassignedActionException If the action could not be assigned to any resource.
     */
    public final void scheduleAndLaunchAction(AllocatableAction aa, Score score)
        throws BlockedActionException, UnassignedActionException {
        aa.schedule(score);
        tryToLaunch(aa);
    }
}