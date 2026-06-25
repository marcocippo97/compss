package es.bsc.compss.scheduler.rank.prediction.types;

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
