package es.bsc.compss.scheduler.rank.prediction;

import es.bsc.compss.COMPSsConstants;
import es.bsc.compss.components.impl.ResourceScheduler;
import es.bsc.compss.scheduler.exceptions.BlockedActionException;
import es.bsc.compss.scheduler.exceptions.UnassignedActionException;
import es.bsc.compss.scheduler.rank.base.RankBaseTS;
import es.bsc.compss.scheduler.rank.prediction.types.PredictionSchedulingData;
import es.bsc.compss.scheduler.rank.prediction.types.SimilarityEngine;
import es.bsc.compss.scheduler.rank.prediction.types.SimilarityFunction;
import es.bsc.compss.scheduler.rank.prediction.types.SuccessorHint;
import es.bsc.compss.scheduler.rank.prediction.types.TaskFeatures;
import es.bsc.compss.scheduler.rank.prediction.types.TaskGraphCache;
import es.bsc.compss.scheduler.rank.prediction.types.WLSimilarityFunction;
import es.bsc.compss.scheduler.types.ActionOrchestrator;
import es.bsc.compss.scheduler.types.AllocatableAction;
import es.bsc.compss.scheduler.types.ObjectValue;
import es.bsc.compss.scheduler.types.Profile;
import es.bsc.compss.scheduler.types.Score;
import es.bsc.compss.types.allocatableactions.ExecutionAction;
import es.bsc.compss.types.implementations.ExecType;
import es.bsc.compss.types.implementations.Implementation;
import es.bsc.compss.types.implementations.ImplementationDescription;
import es.bsc.compss.types.implementations.definition.ImplementationDefinition;
import es.bsc.compss.types.parameter.impl.BasicTypeParameter;
import es.bsc.compss.types.parameter.impl.CollectiveParameter;
import es.bsc.compss.types.parameter.impl.FileParameter;
import es.bsc.compss.types.parameter.impl.Parameter;
import es.bsc.compss.types.resources.MethodResourceDescription;
import es.bsc.compss.types.resources.Worker;
import es.bsc.compss.types.resources.WorkerResourceDescription;
import es.bsc.compss.types.resources.components.Processor;

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

import org.json.JSONObject;


/**
 * Task scheduler that extends {@link RankBaseTS} with a similarity-based look-ahead scheduling policy.
 */
public class PredictionTS extends RankBaseTS {

    // -------------------------------------------------------------------------
    // Tuning constants
    // -------------------------------------------------------------------------

    private static final int DEFAULT_DEPTH_WINDOW = 3;
    private static final int DEFAULT_MAX_RECENT_WINDOW_SIZE = 1_000;
    private static final double DEFAULT_SIMILARITY_THRESHOLD = 0.7;
    private static final double DEFAULT_PAIRING_SIMILARITY_THRESHOLD = 0.7;
    private static final int DEFAULT_HOPS = 2;
    private static final double DEFAULT_R = 0.8;
    private static final double DEFAULT_ALPHA = 1.0;
    private static final int DEFAULT_MIN_SAMPLES_FOR_NORMALIZATION =
        SimilarityEngine.DEFAULT_MIN_SAMPLES_FOR_NORMALIZATION;
    /**
     * Width of the DAG-depth window used to filter comparison candidates. A task {@code taskOld} is eligible for
     * comparison only when {@code |depth(taskOld) - depth(newTask)| <= DEPTH_WINDOW}.
     */
    private final int depthWindow;

    /**
     * Upper bound on the number of tasks retained in the in-memory sliding window.
     */
    private final int maxRecentWindowSize;

    /**
     * Minimum similarity score required for hint generation. Tasks whose score falls below this threshold are silently
     * ignored.
     */
    private final double similarityThreshold;

    /**
     * Minimum similarity score required for pairing task with a hint.
     */
    private final double pairingSimilarityThreshold;

    // -------------------------------------------------------------------------
    // Similarity look-ahead fields
    // -------------------------------------------------------------------------

    /**
     * Sliding window of recently seen execution tasks, bounded by {@link #maxRecentWindowSize}. Entries are evicted
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

    /**
     * Configuration for adding implementations to prioritized tasks.
     */
    private final Map<Integer, Map<String, ?>> implementationsMap;

    /**
     * Set of already added prioritized implementations.
     */
    private final Set<String> implSet = new HashSet<String>();

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

        String schedulerConfigPath = System.getProperty(COMPSsConstants.SCHEDULER_CONFIG_FILE);
        PredictionSchedulingData predSchedulerConfig = new PredictionSchedulingData(schedulerConfigPath);
        this.implementationsMap = predSchedulerConfig.getImplementations();

        Map<String, Object> params = new HashMap<>();
        for (Map.Entry<String, ?> e : predSchedulerConfig.getParameters().entrySet()) {
            params.put(normalizeKey(e.getKey()), e.getValue());
        }

        this.depthWindow = getIntParam(params, "depthwindow", DEFAULT_DEPTH_WINDOW);
        this.maxRecentWindowSize = getIntParam(params, "maxrecentwindowsize", DEFAULT_MAX_RECENT_WINDOW_SIZE);
        this.similarityThreshold = getDoubleParam(params, "similaritythreshold", DEFAULT_SIMILARITY_THRESHOLD);
        this.pairingSimilarityThreshold =
            getDoubleParam(params, "pairingsimilaritythreshold", DEFAULT_PAIRING_SIMILARITY_THRESHOLD);

        int minSamples = getIntParam(params, "minsamplesfornormalization", DEFAULT_MIN_SAMPLES_FOR_NORMALIZATION);
        int hops = getIntParam(params, "hops", DEFAULT_HOPS);
        double r = getDoubleParam(params, "r", DEFAULT_R);
        double alpha = getDoubleParam(params, "alpha", DEFAULT_ALPHA);

        this.taskGraphCache = new TaskGraphCache();
        this.similarityEngine = new SimilarityEngine(taskGraphCache, minSamples);
        this.similarityFunction = new WLSimilarityFunction(taskGraphCache, similarityEngine, hops, r, alpha);
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

            if (recentTaskWindow.size() >= maxRecentWindowSize) {
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
     * {@link SuccessorHint} is created and stored in {@code successorHintMap}, enabling the hint-based scheduling logic
     * in {@link #handleDependencyFreeActions} to use it in future cycles.
     *
     * @param action The action that has just completed.
     */
    @Override
    public void actionCompleted(AllocatableAction action) {
        if (action.getCoreId() != null) {
            Profile profile = action.getProfile();
            Map<Long, Double> pending =
                taskGraphCache.updateOnCompletion(action.getId(), action.getAssignedResource(), profile);

            TaskGraphCache.NodeInfo succInfo = taskGraphCache.getNode(action.getId());
            if (succInfo != null) {
                for (Map.Entry<Long, Double> entry : pending.entrySet()) {
                    AllocatableAction newTaskAction = taskIdToAction.get(entry.getKey());
                    if (newTaskAction != null) {
                        insertOrUpdateHint(newTaskAction, succInfo.taskId, succInfo.features.getCoreId(),
                            succInfo.executionProfile, entry.getValue());
                    }
                }
            }
        }
        super.actionCompleted(action);
    }

    private void insertOrUpdateHint(AllocatableAction newTaskAction, long candidateTaskId, int coreId, Profile profile,
        double confidence) {

        List<SuccessorHint> siblings = successorHintMap.computeIfAbsent(newTaskAction, k -> new ArrayList<>());

        long consumedCount = siblings.stream().filter(SuccessorHint::isConsumed).count();
        List<SuccessorHint> unconsumed = new ArrayList<>();
        for (SuccessorHint h : siblings) {
            if (!h.isConsumed()) {
                unconsumed.add(h);
            }
        }
        unconsumed.add(new SuccessorHint(coreId, candidateTaskId, profile, confidence, -1));

        unconsumed.sort(Comparator.comparingLong((SuccessorHint h) -> h.getProfile().getExecutionTime()).reversed());
        for (int i = 0; i < unconsumed.size(); i++) {
            unconsumed.get(i).setRank((int) consumedCount + i);
        }

        siblings.removeIf(h -> !h.isConsumed());
        siblings.addAll(unconsumed);
    }

    // -------------------------------------------------------------------------
    // Similarity evaluation and hint generation
    // -------------------------------------------------------------------------

    /**
     * For each task in the sliding window that passes the depth-window and sibling filters, computes a similarity score
     * against {@code taskNew}. When the score exceeds {@link #similarityThreshold}, retrieves the direct successors of
     * the matched task: successors with a known execution profile are sorted by descending average execution time and
     * stored as ranked {@link SuccessorHint}s in {@code successorHintMap[taskNew]}; successors that have not yet
     * completed register a pending-hint request to be fulfilled in {@link #actionCompleted}.
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
            if (oldInfo == null || Math.abs(oldInfo.depth - newInfo.depth) > depthWindow) {
                continue;
            }
            if (!newPredSet.isEmpty() && newPredSet.equals(new HashSet<>(oldInfo.predecessorIds))) {
                continue;
            }
            double sim = similarityFunction.compute(taskNew.getId(), taskOld.getId());
            if (sim <= similarityThreshold) {
                continue;
            }
            LOGGER.debug(
                getLoggerPrefix() + " Similarity match: taskNew=" + taskNew + " taskOld=" + taskOld + " score=" + sim);

            for (TaskGraphCache.NodeInfo succ : taskGraphCache.getSuccessors(taskOld.getId())) {
                if (succ.executionProfile != null) {
                    insertOrUpdateHint(taskNew, succ.taskId, succ.features.getCoreId(), succ.executionProfile, sim);
                } else {
                    succ.pendingHintRequests.put(taskNew.getId(), sim);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Scheduling operations
    // -------------------------------------------------------------------------

    @Override
    public <T extends WorkerResourceDescription> PredictionRS<T> generateSchedulerForResource(Worker<T> w,
        JSONObject resJSON, JSONObject implJSON) {
        return new PredictionRS<>(w, resJSON, implJSON);
    }

    @Override
    public void customCoreElementsUpdated() {
        for (ResourceScheduler<? extends WorkerResourceDescription> rs : this.getWorkers()) {
            ((PredictionRS<?>) rs).updateCoreElements();
        }
    }

    /**
     * Handles actions that have just become free of data dependencies by determining the rank for each action and
     * embedding it into the scheduling score via {@link generateActionScore}.
     */
    @Override
    protected <T extends WorkerResourceDescription> void handleDependencyFreeActions(
        List<AllocatableAction> dataFreeActions, List<AllocatableAction> resourceFreeActions,
        List<AllocatableAction> blockedCandidates, ResourceScheduler<T> resource) {

        manageUpgradedActions(resource);
        PriorityQueue<ObjectValue<AllocatableAction>> executableActions = new PriorityQueue<>();

        for (AllocatableAction freeAction : dataFreeActions) {
            int rank = 0;

            if (freeAction instanceof ExecutionAction) {
                SuccessorHint matchingHint = null;
                AllocatableAction hintPredecessor = null;
                double bestSimilarity = -1.0;

                for (TaskGraphCache.NodeInfo predTask : taskGraphCache.getPredecessors(freeAction.getId())) {
                    AllocatableAction predecessor = taskIdToAction.get(predTask.taskId);
                    List<SuccessorHint> hints = successorHintMap.get(predecessor);
                    if (hints == null) {
                        continue;
                    }
                    for (SuccessorHint hint : hints) {
                        if (hint.isConsumed() || hint.getCoreId() != freeAction.getCoreId()) {
                            continue;
                        }
                        double sim = similarityFunction.compute(freeAction.getId(), hint.getCandidateTaskId());
                        if (sim > bestSimilarity) {
                            bestSimilarity = sim;
                            matchingHint = hint;
                            hintPredecessor = predecessor;
                        }
                    }
                }
                // discard a matching hint if the pairing is not good enough
                if (matchingHint != null && bestSimilarity >= this.pairingSimilarityThreshold) {
                    rank = matchingHint.getRank();
                    matchingHint.markConsumed();
                    // Remove the predecessor's hint list when all sibling hints are consumed.
                    boolean allConsumed =
                        successorHintMap.get(hintPredecessor).stream().allMatch(SuccessorHint::isConsumed);
                    if (allConsumed) {
                        successorHintMap.remove(hintPredecessor);
                    }
                    LOGGER.debug(getLoggerPrefix() + " Hint rank " + rank + " assigned to: " + freeAction);

                    // set a prioritized implementation from scheduler configuration file
                    if (!this.implementationsMap.isEmpty() && implementationsMap.containsKey(rank)) {

                        Implementation oldImpl =
                            ((ExecutionAction) freeAction).getCoreElement().getImplementations().get(0);
                        String newSig = oldImpl.getSignature() + Implementation.priorityImplSuffix + "." + rank;

                        if (!implSet.contains(newSig)) {
                            Boolean localProcessing = oldImpl.isLocalProcessing();

                            ImplementationDescription<?, ?> implDescription = oldImpl.getDescription();
                            ImplementationDefinition implDefinition = implDescription.getDefinition();

                            MethodResourceDescription oldConstraints =
                                (MethodResourceDescription) implDescription.getConstraints();
                            Map<String, ?> overrides = implementationsMap.get(rank);
                            MethodResourceDescription newConstraints =
                                applyConstraintOverrides(oldConstraints, overrides);

                            ExecType prolog = implDescription.getProlog();
                            ExecType epilog = implDescription.getEpilog();

                            ImplementationDescription<?, ?> newImplDesc = new ImplementationDescription<>(
                                implDefinition, newSig, localProcessing, newConstraints, prolog, epilog);

                            ((ExecutionAction) freeAction).getCoreElement().addImplementation(newImplDesc);
                            coreElementsUpdated();
                            implSet.add(newSig);
                            LOGGER.debug(getLoggerPrefix() + " New priority implementation added: " + newSig);
                        }
                        freeAction.getSchedulingInfo().setPrioritizedImpl(newSig);
                    }
                }
            }
            Score actionScore = generateActionScore(freeAction, rank);
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
    // -------------------------------------------------------------------------
    // Constraint override support (rank-based prioritized implementations)
    // -------------------------------------------------------------------------

    /**
     * Builds a new {@link MethodResourceDescription} by applying {@code overrides} on top of a copy of {@code base}.
     */
    protected MethodResourceDescription applyConstraintOverrides(MethodResourceDescription base,
        Map<String, ?> overrides) {

        MethodResourceDescription result = new MethodResourceDescription(base);

        // Normalize keys once so lookups (here and in selectTargetProcessor) are consistent
        // regardless of the casing/underscore style used in the JSON configuration file.
        Map<String, Object> norm = new HashMap<>();
        for (Map.Entry<String, ?> e : overrides.entrySet()) {
            norm.put(normalizeKey(e.getKey()), e.getValue());
        }

        Processor targetProc = selectTargetProcessor(result, norm);
        boolean isNewProcessor = true;
        for (Processor p : result.getProcessors()) {
            if (p == targetProc) {
                isNewProcessor = false;
                break;
            }
        }
        boolean processorTouched = false;

        for (Map.Entry<String, Object> entry : norm.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();

            switch (key) {
                case "computingunits":
                    targetProc.setComputingUnits(toInt(value));
                    processorTouched = true;
                    break;
                case "processorspeed":
                    targetProc.setSpeed(toFloat(value));
                    processorTouched = true;
                    break;
                case "processorarchitecture":
                    targetProc.setArchitecture(String.valueOf(value));
                    processorTouched = true;
                    break;
                case "processortype":
                    targetProc.setType(String.valueOf(value));
                    processorTouched = true;
                    break;
                case "processorname":
                    targetProc.setName(String.valueOf(value));
                    processorTouched = true;
                    break;
                case "processorinternalmemorysize":
                    targetProc.setInternalMemory(toFloat(value));
                    processorTouched = true;
                    break;
                case "processorpropertyname":
                    targetProc.setPropName(String.valueOf(value));
                    processorTouched = true;
                    break;
                case "processorpropertyvalue":
                    targetProc.setPropValue(String.valueOf(value));
                    processorTouched = true;
                    break;
                case "memorysize":
                    result.setMemorySize(toFloat(value));
                    break;
                case "memorytype":
                    result.setMemoryType(String.valueOf(value));
                    break;
                case "storagesize":
                    result.setStorageSize(toFloat(value));
                    break;
                case "storagetype":
                    result.setStorageType(String.valueOf(value));
                    break;
                case "storagebw":
                    result.setStorageBW(toInt(value));
                    break;
                case "operatingsystemtype":
                    result.setOperatingSystemType(String.valueOf(value));
                    break;
                case "operatingsystemdistribution":
                    result.setOperatingSystemDistribution(String.valueOf(value));
                    break;
                case "operatingsystemversion":
                    result.setOperatingSystemVersion(String.valueOf(value));
                    break;
                case "wallclocklimit":
                    result.setWallClockLimit(toInt(value));
                    break;
                case "appsoftware":
                    result.resetAppSoftware();
                    for (String app : String.valueOf(value).split(",")) {
                        result.addApplication(app.trim());
                    }
                    break;
                case "hostqueues":
                    result.resetHostQueues();
                    for (String q : String.valueOf(value).split(",")) {
                        result.addHostQueue(q.trim());
                    }
                    break;
                default:
                    LOGGER.warn(getLoggerPrefix() + " Unrecognised constraint override key: " + entry.getKey());
            }
        }

        if (processorTouched) {
            if (isNewProcessor) {
                if (targetProc.hasUnassignedCUs()) {
                    targetProc.setComputingUnits(1);
                }
                result.addProcessor(targetProc);
            } else {
                // targetProc was mutated in place: force a full recount of the aggregate
                // counters (totalCPUComputingUnits, totalCPUs, ...), which addProcessor()
                // would not otherwise refresh for an already-registered processor.
                result.setProcessors(result.getProcessors());
            }
        }
        return result;
    }

    /**
     * Selects which processor entry the override map should target: the one matching {@code processorType} if
     * specified, otherwise the first declared processor (or a fresh one if {@code mrd} has none yet).
     *
     * @param mrd The constraints being modified.
     * @param norm Normalized override map.
     * @return The {@link Processor} instance to apply field overrides to.
     */
    private static Processor selectTargetProcessor(MethodResourceDescription mrd, Map<String, Object> norm) {
        Object typeOverride = norm.get("processortype");
        if (typeOverride != null) {
            String wantedType = String.valueOf(typeOverride);
            for (Processor p : mrd.getProcessors()) {
                if (p.getType().name().equalsIgnoreCase(wantedType)) {
                    return p;
                }
            }
            return new Processor();
        }
        List<Processor> procs = mrd.getProcessors();
        return procs.isEmpty() ? new Processor() : procs.get(0);
    }

    // -------------------------------------------------------------------------
    // Utils
    // -------------------------------------------------------------------------

    private static String normalizeKey(String key) {
        return key.toLowerCase().replace("_", "");
    }

    private static int toInt(Object value) {
        return (value instanceof Number) ? ((Number) value).intValue() : Integer.parseInt(String.valueOf(value));
    }

    private static float toFloat(Object value) {
        return (value instanceof Number) ? ((Number) value).floatValue() : Float.parseFloat(String.valueOf(value));
    }

    private static int getIntParam(Map<String, ?> params, String key, int defaultValue) {
        Object v = params.get(key);
        return (v instanceof Number) ? ((Number) v).intValue() : defaultValue;
    }

    private static double getDoubleParam(Map<String, ?> params, String key, double defaultValue) {
        Object v = params.get(key);
        return (v instanceof Number) ? ((Number) v).doubleValue() : defaultValue;
    }

    @Override
    protected String getLoggerPrefix() {
        return "[PredictionTS]";
    }
}