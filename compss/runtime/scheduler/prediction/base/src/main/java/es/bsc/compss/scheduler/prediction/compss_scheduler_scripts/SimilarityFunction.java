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

/**
 * Pluggable strategy for computing the similarity between two tasks, identified by their unique runtime IDs.
 * Implementations receive task IDs rather than AllocatableAction objects so that the interface is decoupled from the
 * scheduler internals; full node metadata is resolved from the {@link TaskGraphCache}.
 */
public interface SimilarityFunction {

    /**
     * Computes the similarity between two tasks.
     *
     * @param taskId1 ID of the first task.
     * @param taskId2 ID of the second task.
     * @return Similarity value in [0.0, 1.0], where 1.0 means identical and 0.0 means completely dissimilar.
     */
    double compute(long taskId1, long taskId2);
}
