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

import es.bsc.compss.scheduler.types.Profile;


/**
 * Data-transfer object carrying a single scheduling suggestion for a task that is predicted to appear as a successor of
 * a newly arrived task. Hints are produced by {@link PredictionTS} when a similarity match is found between a new task
 * ({@code newTask}) and a past task ({@code oldTask}). Each hint describes one direct successor of {@code oldTask}
 * whose historical execution profile is already known, and is associated to {@code newTask} via the
 * {@code successorHintMap}. When a successor of {@code newTask} later becomes dependency-free, its coreId is matched
 * against the hint's {@link #getCoreId()}. If matched, the {@link #getRank()} field is used to decide whether to
 * schedule the action immediately or to defer it until higher-priority siblings have been submitted. The {@code rank}
 * represents the desired scheduling order among the sibling group: rank 0 should be scheduled first. Rank is assigned
 * at hint-creation time by sorting the successors of {@code oldTask} in descending order of their average execution
 * time (Longest Job First), so that the most resource-intensive tasks enter the ready queue earliest and resource
 * utilisation is maximised. The {@code confidence} field equals the similarity score that triggered hint generation
 * (always above {@code SIMILARITY_THRESHOLD}).
 */
public class SuccessorHint {

    private final int coreId;
    private final Profile historicalProfile;
    private final double confidence;

    /**
     * Relative scheduling priority within the sibling group. Lower value = higher priority. Rank 0 is assigned to the
     * successor with the longest expected execution time.
     */
    private final int rank;

    /**
     * Whether this hint has already been matched to a scheduled action. Consumed hints are ignored in subsequent
     * rank-comparison checks.
     */
    private boolean consumed;


    /**
     * Constructs a SuccessorHint.
     *
     * @param coreId Expected coreId of the successor task.
     * @param historicalProfile Execution profile of the analogous past successor (may be {@code null} if not available
     *            at hint-creation time).
     * @param confidence Similarity score that triggered this hint, in [0, 1].
     * @param rank Scheduling priority within the sibling group (0 = highest).
     */
    public SuccessorHint(int coreId, Profile historicalProfile, double confidence, int rank) {
        this.coreId = coreId;
        this.historicalProfile = historicalProfile;
        this.confidence = confidence;
        this.rank = rank;
        this.consumed = false;
    }

    /**
     * Returns the expected coreId of the successor task this hint targets.
     *
     * @return Core element ID.
     */
    public int getCoreId() {
        return coreId;
    }

    /**
     * Returns the historical execution profile of the analogous past successor, or {@code null} if the profile was not
     * available when the hint was built.
     *
     * @return Historical profile, or {@code null}.
     */
    public Profile getHistoricalProfile() {
        return historicalProfile;
    }

    /**
     * Returns the confidence weight of this hint in [0.0, 1.0], equal to the similarity score that triggered hint
     * generation.
     *
     * @return Confidence value.
     */
    public double getConfidence() {
        return confidence;
    }

    /**
     * Returns the scheduling rank of this hint within its sibling group. Lower value means higher priority (should be
     * submitted to the ready queue first).
     *
     * @return Rank index, starting at 0.
     */
    public int getRank() {
        return rank;
    }

    /**
     * Returns whether this hint has already been matched to a scheduled action.
     *
     * @return {@code true} if consumed.
     */
    public boolean isConsumed() {
        return consumed;
    }

    /**
     * Marks this hint as consumed so that it is excluded from subsequent rank-comparison checks.
     */
    public void markConsumed() {
        this.consumed = true;
    }
}
