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
package es.bsc.compss.scheduler.prediction.base;

import es.bsc.compss.components.impl.ResourceScheduler;
import es.bsc.compss.components.impl.TaskScheduler;
import es.bsc.compss.scheduler.exceptions.BlockedActionException;
import es.bsc.compss.scheduler.exceptions.UnassignedActionException;
import es.bsc.compss.scheduler.prediction.base.PredictionSchedulingOptimizer;
import es.bsc.compss.scheduler.prediction.base.SimilarityEngine;
import es.bsc.compss.scheduler.prediction.base.SimilarityFunction;
import es.bsc.compss.scheduler.prediction.base.SuccessorHint;
import es.bsc.compss.scheduler.prediction.base.TaskFeatures;
import es.bsc.compss.scheduler.prediction.base.TaskGraphCache;
import es.bsc.compss.scheduler.prediction.base.WLSimilarityFunction;
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


/**
 * Task scheduler that extends the order-strict scheduling policy with a similarity-based look-ahead mechanism. Overview
 * of the prediction flow: - DEPTH-WINDOW FILTER. When a new task ({@code newTask}) arrives, candidates for comparison
 * are restricted to tasks in the sliding window whose DAG depth satisfies:
 * {@code |depth(taskOld) - depth(newTask)| <= DEPTH_WINDOW}. Direct predecessors of {@code newTask} are excluded
 * because their relationship is already captured structurally. - SIMILARITY SCORING. For each candidate that passes the
 * filter, a multi-level weighted similarity score is computed by {@link WLSimilarityFunction}, comparing categorical
 * features (from the CoreElement) and numerical features at each predecessor level up to {@code hops} levels. - HINT
 * GENERATION. If the score exceeds {@link #SIMILARITY_THRESHOLD}, the {@link TaskGraphCache} is queried for the direct
 * successors of the matched {@code oldTask}: - If a successor has a known execution profile, a {@link SuccessorHint} is
 * created immediately and associated to {@code newTask} in {@code successorHintMap}. - If a successor has not yet
 * completed, a pending-hint request is registered in {@link TaskGraphCache.NodeInfo#pendingHintRequests}. When that
 * successor eventually completes (see {@link #actionCompleted}), the deferred hint is built and added to
 * {@code successorHintMap}. Hints are sorted by the successors' average execution time (Longest Job First) and assigned
 * a rank starting at 0. - HINT CONSUMPTION. When a successor of {@code newTask} becomes dependency-free (in
 * {@link #handleDependencyFreeActions}), its coreId is matched against the hints in {@code successorHintMap[newTask]}:
 * - If the action's hint rank is the lowest among all unconsumed sibling hints, it is submitted to the ready queue
 * immediately. - Otherwise it is placed in {@code deferredActions} and re-evaluated on each subsequent scheduling
 * cycle. After {@link #DEFER_TIMEOUT_MS} milliseconds without a lower-ranked sibling appearing, the action is released
 * unconditionally to prevent starvation.
 */
public class PredictionTS extends TaskScheduler {

    // -------------------------------------------------------------------------
    // Tuning constants
    // -------------------------------------------------------------------------

    /**
     * Half-width of the DAG-depth window used to filter comparison candidates. A task {@code taskOld} is eligible for
     * comparison only when: {@code |depth(taskOld) - depth(newTask)| <= DEPTH_WINDOW}.
     */
    private static final int DEPTH_WINDOW = 3;

    /**
     * Upper bound on the number of tasks retained in the in-memory sliding window. Acts as a memory safety limit
     * independent of the depth filter.
     */
    private static final int MAX_RECENT_WINDOW_SIZE = 1_000;

    /**
     * Minimum similarity score required for hint generation. Tasks whose score falls below this threshold are silently
     * ignored.
     */
    private static final double SIMILARITY_THRESHOLD = 0.7;

    // -------------------------------------------------------------------------
    // Order-strict scheduler fields
    // -------------------------------------------------------------------------

    /**
     * Priority queue of ready actions sorted by scheduling score.
     */
    protected final PriorityQueue<ObjectValue<AllocatableAction>> readyQueue;

    /**
     * Actions that have been upgraded and must be handled with priority.
     */
    protected Set<AllocatableAction> upgradedActions;

    /**
     * Map from action to its current ObjectValue wrapper in the ready queue.
     */
    protected final Map<AllocatableAction, ObjectValue<AllocatableAction>> addedActions;

    // -------------------------------------------------------------------------
    // Similarity look-ahead fields
    // -------------------------------------------------------------------------

    /**
     * Sliding window of recently seen execution tasks, bounded by {@link #MAX_RECENT_WINDOW_SIZE}. Entries are evicted
     * FIFO when the bound is reached.
     */
    private final Deque<AllocatableAction> recentTaskWindow = new ArrayDeque<>();

    /**
     * Reverse map from task ID to the corresponding AllocatableAction, used when processing pending hint requests after
     * a previously unexecuted successor completes.
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
     * {@code oldTask}'s successors. Entries are consumed in {@link #handleDependencyFreeActions} when a predicted
     * successor becomes dependency-free, and removed once all hints in the list have been consumed.
     */
    private final Map<AllocatableAction, List<SuccessorHint>> successorHintMap = new HashMap<>();

    /**
     * Actions that have been deferred because a lower-ranked sibling has not yet appeared. Maps deferred action to the
     * timestamp (in ms) at which it was deferred. Actions are released unconditionally after {@link #DEFER_TIMEOUT_MS}.
     */
    final Map<AllocatableAction, Long> deferredActions = new ConcurrentHashMap<>();

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
        LOGGER.debug("[PredictionTS] Initialising PredictionTS");

        this.readyQueue = new PriorityQueue<>();
        this.upgradedActions = new HashSet<>();
        this.addedActions = new HashMap<>();

        this.taskGraphCache = new TaskGraphCache();
        this.similarityEngine = new SimilarityEngine(taskGraphCache);
        this.similarityFunction =
            new WLSimilarityFunction(taskGraphCache, similarityEngine, /* hops */ 2, /* r */ 0.8, /* alpha */ 0.0);
    }

    // TODO: override generateSchedulingInformation to return PredictionSchedulingInformation
    // once the hint-transfer flow is fully integrated.

    @Override
    public void upgradeAction(AllocatableAction action) {
        if (DEBUG) {
            LOGGER.debug("[PredictionTS] Upgrading action " + action);
        }
        upgradedActions.add(action);
        ObjectValue<AllocatableAction> obj = addedActions.remove(action);
        readyQueue.remove(obj);
    }

    // -------------------------------------------------------------------------
    // Entry point: new task arrival
    // -------------------------------------------------------------------------

    /**
     * Entry point for every new action that enters the scheduler. For execution actions the method: - Registers the
     * task in the {@link TaskGraphCache} with its features and predecessor links, then updates the
     * {@link SimilarityEngine} feature space. - Evaluates similarity against depth-window candidates and builds hints
     * if a match is found. - Adds the action to the sliding window (after evaluation, to avoid self-comparison). -
     * Proceeds with normal order-strict scheduling via {@code super}. Non-execution actions (system actions) bypass
     * steps 1–3 and are forwarded directly to {@code super}.
     */
    @Override
    public final void scheduleAction(AllocatableAction action, Score actionScore) throws BlockedActionException {
        // Only execution actions carry task features and are relevant for similarity matching
        if (action instanceof ExecutionAction) {

            ExecutionAction execAction = (ExecutionAction) action;
            int coreId = execAction.getCoreId();

            // Build categorical features from task and parameter names.
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
                        ;// TODO to insert a method to retrieve file size
                    }
                }
            }

            TaskFeatures features = new TaskFeatures(coreId, categorical, numerical);

            // Collect predecessor IDs from actions already registered in the cache.
            List<Long> predecessorIds = new ArrayList<>();
            // TODO if a synchronization point is set in the graph, the dependency relation is lost
            for (AllocatableAction pred : execAction.getDataPredecessors()) {
                predecessorIds.add(pred.getId());
            }

            taskGraphCache.registerTask(execAction.getId(), features, predecessorIds);
            taskIdToAction.put(execAction.getId(), execAction);

            // Update the feature index with this node's numerical features.
            similarityEngine.updateFeatureSpace(execAction.getId());

            // Evaluate similarity and generate hints before adding to the window.
            evaluateSimilarityAndBuildHints(execAction);

            // Add to window after evaluation to prevent self-comparison.
            if (recentTaskWindow.size() >= MAX_RECENT_WINDOW_SIZE) {
                AllocatableAction evicted = recentTaskWindow.pollFirst();
                successorHintMap.remove(evicted);
            }
            recentTaskWindow.addLast(execAction);
        }

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

    /**
     * Called when an action has finished executing. Updates the {@link TaskGraphCache} with the assigned resource and
     * execution profile, then processes any pending hint requests that were registered while the node was still
     * unexecuted. For each pending request a {@link SuccessorHint} is built and added to {@code successorHintMap} keyed
     * by the corresponding {@code newTask}, enabling the rank-based scheduling logic in
     * {@link #handleDependencyFreeActions} to use this hint in future scheduling cycles.
     *
     * @param action The action that has just completed.
     */
    @Override
    public void actionCompleted(AllocatableAction action) {

        if (action.getCoreId() != null) {
            ResourceScheduler<?> rs = action.getAssignedResource();
            // Profile profile = rs.getProfile(action.getAssignedImplementation());
            Profile profile = action.getProfile();

            Map<Long, Double> pending = taskGraphCache.updateOnCompletion(action.getId(), rs, profile);

            for (Map.Entry<Long, Double> entry : pending.entrySet()) {
                long newTaskId = entry.getKey();
                double confidence = entry.getValue();
                AllocatableAction newTaskAction = taskIdToAction.get(newTaskId);
                TaskGraphCache.NodeInfo succInfo = taskGraphCache.getNode(action.getId());

                if (newTaskAction != null && succInfo != null) {
                    // Rank is 0 for deferred hints because no sibling comparison was possible
                    // at the time of registration.
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
     * For each task in the sliding window that passes the depth-window filter, computes the multi-level similarity
     * score against {@code taskNew}. When the score exceeds {@link #SIMILARITY_THRESHOLD}, retrieves the direct
     * successors of the matched {@code oldTask} from the {@link TaskGraphCache} and processes them: - Successors whose
     * execution profile is already known are sorted by descending average execution time (Longest Job First) and
     * assigned ranks 0, 1, 2, … A {@link SuccessorHint} is created for each and stored in
     * {@code successorHintMap[taskNew]}. - Successors that have not yet completed have their {@code taskNew} ID
     * registered in {@link TaskGraphCache.NodeInfo#pendingHintRequests} so that a hint can be built retroactively when
     * they complete (see {@link #actionCompleted}). Filter rules applied in order: - Skip direct predecessors of
     * {@code taskNew}. - Skip tasks whose DAG depth differs from that of {@code taskNew} by more than
     * {@link #DEPTH_WINDOW}.
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

            // Rule 1: skip direct predecessors of taskNew.
            if (newInfo.predecessorIds.contains(taskOld.getId())) {
                continue;
            }

            // Rule 2: depth-window filter.
            TaskGraphCache.NodeInfo oldInfo = taskGraphCache.getNode(taskOld.getId());
            if (oldInfo == null) {
                continue;
            }
            if (Math.abs(oldInfo.depth - depthNew) > DEPTH_WINDOW) {
                continue;
            }

            // Rule 3: skip siblings (same non-empty predecessor set).
            Set<Long> oldPredSet = new HashSet<>(oldInfo.predecessorIds);
            if (!newPredSet.isEmpty() && newPredSet.equals(oldPredSet)) {
                continue;
            }

            double sim = similarityFunction.compute(taskNew.getId(), taskOld.getId());

            if (sim > SIMILARITY_THRESHOLD) {
                LOGGER.debug(
                    "[PredictionTS] Similarity match: taskNew=" + taskNew + " taskOld=" + taskOld + " score=" + sim);

                List<TaskGraphCache.NodeInfo> successors = taskGraphCache.getSuccessors(taskOld.getId());

                // Separate successors with a known profile from those still pending.
                List<TaskGraphCache.NodeInfo> completedSuccessors = new ArrayList<>();
                for (TaskGraphCache.NodeInfo succ : successors) {
                    if (succ.executionProfile != null) {
                        completedSuccessors.add(succ);
                    } else {
                        // Register a pending hint request on the unexecuted successor node so
                        // that actionCompleted() can build the hint retroactively once the
                        // successor's execution profile becomes available.
                        succ.pendingHintRequests.put(taskNew.getId(), sim);
                    }
                }

                // Sort completed successors by descending average execution time (LJF order)
                // and assign rank 0, 1, 2, … so the longest task enters the queue first.
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
    // Scheduling operations (order-strict logic + hint consumption)
    // -------------------------------------------------------------------------

    private PriorityQueue<ObjectValue<AllocatableAction>> sortActions(Iterable<AllocatableAction> actions) {
        if (DEBUG) {
            LOGGER.debug("[PredictionTS] Sorting upgraded actions.");
        }
        PriorityQueue<ObjectValue<AllocatableAction>> sorted = new PriorityQueue<>();
        for (AllocatableAction action : actions) {
            Score score = generateActionScore(action);
            sorted.add(new ObjectValue<>(action, score));
        }
        return sorted;
    }

    private void manageUpgradedActions(ResourceScheduler<?> resource) {
        if (upgradedActions.isEmpty()) {
            return;
        }
        if (DEBUG) {
            LOGGER.debug("[PredictionTS] Managing " + upgradedActions.size() + " upgraded actions.");
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

    private void addActionToReadyQueue(AllocatableAction action, Score actionScore) {
        ObjectValue<AllocatableAction> obj = new ObjectValue<>(action, actionScore);
        addedActions.put(action, obj);
        readyQueue.add(obj);
    }

    /**
     * Returns the set of predecessor task IDs for the given action as recorded in the {@link TaskGraphCache}.
     */
    Set<Long> getPredecessorIds(AllocatableAction action) {
        Set<Long> predIds = new HashSet<>();
        for (TaskGraphCache.NodeInfo pred : taskGraphCache.getPredecessors(action.getId())) {
            predIds.add(pred.taskId);
        }
        return predIds;
    }

    /**
     * Handles tasks that have just become free of data or resource dependencies. For each newly data-free action the
     * method first checks whether the action has a matching {@link SuccessorHint} in {@code successorHintMap}: - If the
     * hint's {@link SuccessorHint#getRank()} is the lowest among all unconsumed sibling hints for the same predecessor,
     * the action is submitted to the executable queue immediately and the hint is marked consumed. - Otherwise the
     * action is placed in {@code deferredActions} and re-evaluated on every subsequent call. After
     * {@link #DEFER_TIMEOUT_MS} milliseconds the action is released unconditionally to prevent starvation. - If no
     * matching hint is found, the action is scheduled immediately following the standard order-strict logic. A
     * predecessor's entry in {@code successorHintMap} is removed once all its hints have been consumed.
     */
    @Override
    protected <T extends WorkerResourceDescription> void handleDependencyFreeActions(
        List<AllocatableAction> dataFreeActions, List<AllocatableAction> resourceFreeActions,
        List<AllocatableAction> blockedCandidates, ResourceScheduler<T> resource) {

        manageUpgradedActions(resource);

        PriorityQueue<ObjectValue<AllocatableAction>> executableActions = new PriorityQueue<>();

        // Process newly data-free actions.
        for (AllocatableAction freeAction : dataFreeActions) {

            SuccessorHint matchingHint = null;
            AllocatableAction hintPredecessor = null;

            // Search predecessors for a hint whose coreId matches this action.
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
                // Determine whether any unconsumed sibling hint has a lower rank.
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
                    // This action has the highest priority among its sibling group: submit now.
                    matchingHint.markConsumed();
                    Score actionScore = generateActionScore(freeAction);
                    executableActions.add(new ObjectValue<>(freeAction, actionScore));
                    LOGGER.debug("[PredictionTS] Hint rank " + freeActionRank + " is lowest — scheduling immediately: "
                        + freeAction);
                } else {
                    // A higher-priority sibling is expected: defer this action.
                    deferredActions.put(freeAction, System.currentTimeMillis());
                    LOGGER.debug("[PredictionTS] Hint rank " + freeActionRank
                        + " deferred — awaiting lower-ranked sibling: " + freeAction);
                }

                // Remove predecessor's entry from the hint map once all hints are consumed.
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
                // No hint available: fall back to Task-rank-based sibling ordering
                // for ExecutionActions whose rank is non-zero.
                if (freeAction instanceof ExecutionAction) {
                    ExecutionAction execFreeAction = (ExecutionAction) freeAction;
                    int taskRank = execFreeAction.getTask().getRank();

                    if (taskRank != 0) {
                        // Determine the predecessor-ID set of this action to identify siblings
                        // (actions that share at least one common predecessor).
                        Set<Long> freeActionPredIds = getPredecessorIds(freeAction);
                        boolean hasLowerRankedSibling = false;

                        // Check among the current batch of newly data-free actions.
                        for (AllocatableAction other : dataFreeActions) {
                            if (other == freeAction || !(other instanceof ExecutionAction)) {
                                continue;
                            }
                            int otherRank = ((ExecutionAction) other).getTask().getRank();
                            if (otherRank >= taskRank) {
                                continue; // not a lower-ranked candidate
                            }
                            // Two actions are siblings when they share at least one predecessor.
                            Set<Long> otherPredIds = getPredecessorIds(other);
                            if (!java.util.Collections.disjoint(freeActionPredIds, otherPredIds)) {
                                hasLowerRankedSibling = true;
                                break;
                            }
                        }
                        // If not found yet, also check already-deferred actions.
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
                            // A sibling with higher scheduling priority exists: defer.
                            deferredActions.put(freeAction, System.currentTimeMillis());
                            LOGGER.debug("[PredictionTS] Task rank " + taskRank
                                + " deferred (rank-based) — awaiting lower-ranked sibling: " + freeAction);
                        } else {
                            // This action has the lowest rank among its live siblings: schedule now.
                            Score actionScore = generateActionScore(freeAction);
                            executableActions.add(new ObjectValue<>(freeAction, actionScore));
                            LOGGER.debug("[PredictionTS] Task rank " + taskRank
                                + " is lowest available (rank-based) — scheduling immediately: " + freeAction);
                        }

                    } else {
                        // rank == 0 means highest priority: schedule immediately, no sibling check needed.
                        Score actionScore = generateActionScore(freeAction);
                        executableActions.add(new ObjectValue<>(freeAction, actionScore));
                    }
                } else {
                    // Non-ExecutionAction (system action): schedule immediately, no rank logic.
                    Score actionScore = generateActionScore(freeAction);
                    executableActions.add(new ObjectValue<>(freeAction, actionScore));
                }
            }
        }
        // No resourceFreeActions handled in this scheduler variant.

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

    /**
     * schedules and tries to launch an action.
     */
    public final void scheduleAndLaunchAction(AllocatableAction aa, Score score)
        throws BlockedActionException, UnassignedActionException {

        aa.schedule(score);
        tryToLaunch(aa);
    }

    /**
     * generate a new PredictionSchedulingOptimizer.
     */
    @Override
    @SuppressWarnings("unchecked")
    public PredictionSchedulingOptimizer generateSchedulingOptimizer() {
        return (PredictionSchedulingOptimizer) new PredictionSchedulingOptimizer(this);
    }
}
