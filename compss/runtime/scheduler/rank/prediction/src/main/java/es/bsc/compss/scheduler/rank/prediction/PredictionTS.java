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
 * fallback scheduling policy.
 */
public class PredictionTS extends RankBaseTS {

    // -------------------------------------------------------------------------
    // Tuning constants
    // -------------------------------------------------------------------------

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

    /**
     * Shared task-graph cache providing DAG structure and per-node features.
     */
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
     * {@code oldTask}'s successors. Hints are consumed in {@link #handleDependencyFreeActions} when a predicted
     * successor becomes dependency-free, and the entry is removed once all hints in the list have been consumed.
     */
    private final Map<AllocatableAction, List<SuccessorHint>> successorHintMap = new HashMap<>();

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

    @Override
    protected String getLoggerPrefix() {
        return "[PredictionTS]";
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

            int counter = 1;
            for (Parameter p : execAction.getTask().getParameters()) {
                categorical.put("param" + counter++, p.getName());
                if (!p.isPotentialDependency()) {
                    if (p instanceof BasicTypeParameter) {
                        BasicTypeParameter sp = (BasicTypeParameter) p;
                        if (sp.getValue() instanceof Number) {
                            numerical.put(sp.getName(), new double[] { ((Number) sp.getValue()).doubleValue() });
                        }
                    } else if (p instanceof CollectiveParameter) {
                        numerical.put(p.getName(),
                            new double[] { (double) ((CollectiveParameter) p).getElements().size() });
                    } else if (p instanceof FileParameter) {
                        ; // TODO: add method to retrieve file size
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
                successorHintMap.remove(recentTaskWindow.pollFirst());
            }
            recentTaskWindow.addLast(execAction);
        }
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
            Profile profile = action.getProfile();
            Map<Long, Double> pending =
                taskGraphCache.updateOnCompletion(action.getId(), action.getAssignedResource(), profile);

            for (Map.Entry<Long, Double> entry : pending.entrySet()) {
                AllocatableAction newTaskAction = taskIdToAction.get(entry.getKey());
                TaskGraphCache.NodeInfo succInfo = taskGraphCache.getNode(action.getId());
                if (newTaskAction != null && succInfo != null) {
                    SuccessorHint hint = new SuccessorHint(succInfo.features.getCoreId(), succInfo.executionProfile,
                        entry.getValue(), 0);
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
        Set<Long> newPredSet = new HashSet<>(newInfo.predecessorIds);

        for (AllocatableAction taskOld : recentTaskWindow) {
            if (newInfo.predecessorIds.contains(taskOld.getId())) {
                continue;
            }
            TaskGraphCache.NodeInfo oldInfo = taskGraphCache.getNode(taskOld.getId());
            if (oldInfo == null || Math.abs(oldInfo.depth - newInfo.depth) > DEPTH_WINDOW) {
                continue;
            }
            if (!newPredSet.isEmpty() && newPredSet.equals(new HashSet<>(oldInfo.predecessorIds))) {
                continue;
            }

            double sim = similarityFunction.compute(taskNew.getId(), taskOld.getId());
            if (sim <= SIMILARITY_THRESHOLD) {
                continue;
            }

            LOGGER.debug(
                getLoggerPrefix() + " Similarity match: taskNew=" + taskNew + " taskOld=" + taskOld + " score=" + sim);

            List<TaskGraphCache.NodeInfo> completed = new ArrayList<>();
            for (TaskGraphCache.NodeInfo succ : taskGraphCache.getSuccessors(taskOld.getId())) {
                if (succ.executionProfile != null) {
                    completed.add(succ);
                } else {
                    succ.pendingHintRequests.put(taskNew.getId(), sim);
                }
            }

            completed.sort(Comparator
                .comparingLong((TaskGraphCache.NodeInfo s) -> s.executionProfile.getExecutionTime()).reversed());

            for (int rankIdx = 0; rankIdx < completed.size(); rankIdx++) {
                TaskGraphCache.NodeInfo succ = completed.get(rankIdx);
                successorHintMap.computeIfAbsent(taskNew, k -> new ArrayList<>())
                    .add(new SuccessorHint(succ.features.getCoreId(), succ.executionProfile, sim, rankIdx));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Scheduling operations
    // -------------------------------------------------------------------------

    /**
     * Handles actions that have just become free of data dependencies by determining the effective rank for each action
     * and embedding it into the scheduling score via {@link generateActionScore}. All actions are submitted to the
     * executable queue immediately; the {@link PriorityQueue} ordering guarantees that actions with lower effective
     * rank (higher priority) are dispatched first when resources are contested.
     */
    @Override
    protected <T extends WorkerResourceDescription> void handleDependencyFreeActions(
        List<AllocatableAction> dataFreeActions, List<AllocatableAction> resourceFreeActions,
        List<AllocatableAction> blockedCandidates, ResourceScheduler<T> resource) {

        manageUpgradedActions(resource);

        PriorityQueue<ObjectValue<AllocatableAction>> executableActions = new PriorityQueue<>();

        for (AllocatableAction freeAction : dataFreeActions) {
            int effectiveRank = 0;

            if (freeAction instanceof ExecutionAction) {
                // --- Tier 1: hint-based rank (siblings only) ---
                SuccessorHint matchingHint = null;
                AllocatableAction hintPredecessor = null;

                for (TaskGraphCache.NodeInfo predTask : taskGraphCache.getPredecessors(freeAction.getId())) {
                    AllocatableAction predecessor = taskIdToAction.get(predTask.taskId);
                    List<SuccessorHint> hints = successorHintMap.get(predecessor);
                    if (hints == null) {
                        continue;
                    }
                    for (SuccessorHint hint : hints) {
                        // TODO: the pairing between actions should follow the same similarity logic
                        // used to create the hints
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
                    effectiveRank = matchingHint.getRank();
                    matchingHint.markConsumed();
                    // Remove the predecessor's hint list when all sibling hints are consumed.
                    boolean allConsumed =
                        successorHintMap.get(hintPredecessor).stream().allMatch(SuccessorHint::isConsumed);
                    if (allConsumed) {
                        successorHintMap.remove(hintPredecessor);
                    }
                    LOGGER.debug(getLoggerPrefix() + " Hint rank " + effectiveRank + " assigned to: " + freeAction);
                } else {
                    // --- Tier 2: task-rank fallback ---
                    effectiveRank = ((ExecutionAction) freeAction).getTask().getRank();
                    LOGGER.debug(
                        getLoggerPrefix() + " Task rank " + effectiveRank + " (fallback) assigned to: " + freeAction);
                }
            }

            Score actionScore = generateActionScore(freeAction, effectiveRank);
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