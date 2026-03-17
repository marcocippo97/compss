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
package es.bsc.compss.scheduler.prediction;

import es.bsc.compss.components.impl.ResourceScheduler;
import es.bsc.compss.scheduler.types.Profile;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


/**
 * Bounded cache of the task DAG seen so far by the runtime. Maintains up to {@link #MAX_CACHE_DEPTH} task nodes with
 * FIFO eviction. For each node it stores the DAG depth, feature container, assigned resource, execution profile, and
 * explicit predecessor/successor adjacency lists. All public methods are {@code synchronized} to allow concurrent
 * access from the scheduler thread.
 */
public class TaskGraphCache {

    /**
     * Maximum number of task nodes retained before the oldest entry is evicted using FIFO order.
     */
    private static final int MAX_CACHE_DEPTH = 5_000;

    /**
     * Main storage: taskId to NodeInfo, with automatic FIFO eviction via removeEldestEntry.
     */
    private final Map<Long, NodeInfo> cache = new LinkedHashMap<Long, NodeInfo>(MAX_CACHE_DEPTH, 0.75f, false) {

        @Override
        protected boolean removeEldestEntry(Map.Entry<Long, NodeInfo> eldest) {
            return size() > MAX_CACHE_DEPTH;
        }
    };


    /**
     * Metadata node stored in the cache for a single task instance.
     */
    public static class NodeInfo {

        /**
         * Unique task identifier, matching {@code AllocatableAction.getId()}.
         */
        public final long taskId;

        /**
         * DAG depth computed as the longest path from any root. depth = 0 for root tasks; depth = 1 + max(pred.depth
         * for pred) otherwise.
         */
        public final int depth;

        /**
         * Categorical and numerical feature container for this node.
         */
        public final TaskFeatures features;

        /**
         * ResourceScheduler on which this task was executed. Remains {@code null} until
         * {@link TaskGraphCache#updateOnCompletion} is called.
         */
        public ResourceScheduler<?> assignedResource;

        /**
         * Execution profile recorded after task completion. Remains {@code null} until
         * {@link TaskGraphCache#updateOnCompletion} is called.
         */
        public Profile executionProfile;

        /**
         * IDs of direct predecessor nodes (one hop back in the DAG).
         */
        public final List<Long> predecessorIds = new ArrayList<>();

        /**
         * IDs of direct successor nodes (one hop forward in the DAG).
         */
        public final List<Long> successorIds = new ArrayList<>();

        /**
         * Pending hint requests accumulated while this node has not yet completed. Maps the ID of a newTask (found
         * similar to the task preceding this one) to the similarity confidence that triggered the request. Processed by
         * the caller inside actionCompleted.
         */
        public final Map<Long, Double> pendingHintRequests = new LinkedHashMap<>();


        NodeInfo(long taskId, int depth, TaskFeatures features) {
            this.taskId = taskId;
            this.depth = depth;
            this.features = features;
        }
    }


    /**
     * Registers a new task node and wires its predecessor/successor edges. Depth is 0 if predecessorIds is empty;
     * otherwise 1 + max(pred.depth for pred).
     *
     * @param taskId Unique identifier of the new task.
     * @param features Feature container built from the task's CoreElement.
     * @param predecessorIds IDs of the tasks that this task directly depends on.
     */
    public synchronized void registerTask(long taskId, TaskFeatures features, List<Long> predecessorIds) {
        int depth = 0;
        for (Long predId : predecessorIds) {
            NodeInfo pred = cache.get(predId);
            if (pred != null) {
                depth = Math.max(depth, pred.depth + 1);
            }
        }

        NodeInfo node = new NodeInfo(taskId, depth, features);
        node.predecessorIds.addAll(predecessorIds);
        cache.put(taskId, node);

        for (Long predId : predecessorIds) {
            NodeInfo pred = cache.get(predId);
            if (pred != null) {
                pred.successorIds.add(taskId);
            }
        }
    }

    /**
     * Records the assigned resource and execution profile of a completed task, and returns any pending hint requests
     * that were registered while the node was still unexecuted. The returned map must be processed by the caller to
     * build deferred {@link SuccessorHint} objects.
     *
     * @param taskId Identifier of the completed task.
     * @param resource ResourceScheduler that executed the task.
     * @param profile Execution profile collected at completion.
     * @return Map from newTaskId to similarity confidence for all pending hint requests; empty if none.
     */
    public synchronized Map<Long, Double> updateOnCompletion(long taskId, ResourceScheduler<?> resource,
        Profile profile) {
        NodeInfo node = cache.get(taskId);
        if (node == null) {
            return Collections.emptyMap();
        }
        node.assignedResource = resource;
        node.executionProfile = profile;

        Map<Long, Double> pending = new LinkedHashMap<>(node.pendingHintRequests);
        node.pendingHintRequests.clear();
        return pending;
    }

    /**
     * Returns the {@link NodeInfo} for the given task, or {@code null} if the node has been evicted.
     *
     * @param taskId Task identifier to look up.
     * @return Corresponding NodeInfo, or {@code null}.
     */
    public synchronized NodeInfo getNode(long taskId) {
        return cache.get(taskId);
    }

    /**
     * Returns task IDs grouped by BFS level, walking backward through predecessor edges up to {@code maxHops} levels
     * from {@code rootId}. Level 0 = {rootId}, level k = predecessors of level k-1. Evicted nodes are silently skipped.
     *
     * @param rootId Starting task ID.
     * @param maxHops Maximum number of predecessor hops to traverse.
     * @return Map from level index to the list of task IDs at that level.
     */
    public synchronized Map<Integer, List<Long>> getNodesByLevel(long rootId, int maxHops) {
        Map<Integer, List<Long>> levels = new HashMap<>();
        levels.put(0, Collections.singletonList(rootId));

        Deque<long[]> queue = new ArrayDeque<>();
        queue.add(new long[] { rootId,
            0 });

        while (!queue.isEmpty()) {
            long[] entry = queue.poll();
            long cur = entry[0];
            int level = (int) entry[1];
            if (level >= maxHops) {
                continue;
            }
            NodeInfo node = cache.get(cur);
            if (node == null) {
                continue;
            }
            for (Long predId : node.predecessorIds) {
                levels.computeIfAbsent(level + 1, k -> new ArrayList<>()).add(predId);
                queue.add(new long[] { predId,
                    level + 1 });
            }
        }
        return levels;
    }

    /**
     * Returns the {@link NodeInfo} objects of the direct successors of {@code rootId} (one hop forward). Used to
     * extract scheduling hints after a similarity match is found.
     *
     * @param rootId Starting task ID.
     * @return List of NodeInfo objects for all direct successors; empty if none are found.
     */
    public synchronized List<NodeInfo> getSuccessors(long rootId) {
        List<NodeInfo> result = new ArrayList<>();
        NodeInfo node = cache.get(rootId);
        if (node == null) {
            return result;
        }
        for (Long succId : node.successorIds) {
            NodeInfo succ = cache.get(succId);
            if (succ != null) {
                result.add(succ);
            }
        }
        return result;
    }
}
