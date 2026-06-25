package es.bsc.compss.scheduler.rank.prediction.types;

import java.util.Collections;
import java.util.List;
import java.util.Map;


/**
 * Default implementation of {@link SimilarityFunction}. Computes a multi-level, geometrically weighted similarity
 * between two tasks by comparing their respective predecessor-level subgraphs up to a configurable depth
 * {@code numLevels}. At each level {@code i} the comparison uses {@link SimilarityEngine#levelSimilarity}, which
 * combines Jaccard similarity on categorical hashes with an optimal-matching score on numerical features. Levels are
 * weighted by {@code r^i}, normalised to sum to 1, so that the immediate context of each task receives proportionally
 * more weight when {@code r < 1}. Parameters: - {@code numLevels} — number of predecessor levels to consider (default
 * 2). numLevels = 0 compares only the task nodes themselves. - {@code r} — geometric decay factor for level weights
 * (default 0.8). r = 1.0 gives uniform weight to all levels. - {@code alpha} — exponent controlling the
 * numerical-dissimilarity penalty within each level (default 0.0, i.e. linear penalty).
 */
public class WLSimilarityFunction implements SimilarityFunction {

    private final int numLevels;
    private final double r;
    private final double alpha;
    private final TaskGraphCache cache;
    private final SimilarityEngine engine;


    /**
     * Constructs a WLSimilarityFunction.
     *
     * @param cache Shared task-graph cache.
     * @param engine Shared similarity engine.
     * @param numLevels Number of predecessor levels to include (>= 0).
     * @param r Geometric decay factor for level weights (0 < r <= 1).
     * @param alpha Numerical-dissimilarity penalty exponent (>= 0).
     */
    public WLSimilarityFunction(TaskGraphCache cache, SimilarityEngine engine, int numLevels, double r, double alpha) {
        this.cache = cache;
        this.engine = engine;
        this.numLevels = numLevels;
        this.r = r;
        this.alpha = alpha;
    }

    /**
     * Computes the multi-level weighted similarity between two tasks. Algorithm: - Retrieve the predecessor-level maps
     * for both tasks up to depth numLevels. - Compute unnormalised weights w[i] = r^i for i in [0..numLevels]. - For
     * each level i, compute {@link SimilarityEngine#levelSimilarity} on the two node lists at that level and accumulate
     * w[i] * score. - Divide the total by the sum of weights to normalise. Level 0 always contains the two task nodes
     * themselves, so their categorical and numerical features are always included in the score.
     *
     * @param taskId1 ID of the first task.
     * @param taskId2 ID of the second task.
     * @return Weighted similarity score in [0.0, 1.0].
     */
    @Override
    public double compute(long taskId1, long taskId2) {
        Map<Integer, List<Long>> levels1 = cache.getNodesByLevel(taskId1, numLevels);
        Map<Integer, List<Long>> levels2 = cache.getNodesByLevel(taskId2, numLevels);

        double[] weights = new double[numLevels + 1];
        double norm = 0.0;
        for (int i = 0; i <= numLevels; i++) {
            weights[i] = Math.pow(r, i);
            norm += weights[i];
        }

        double total = 0.0;
        for (int i = 0; i <= numLevels; i++) {
            List<Long> l1 = levels1.getOrDefault(i, Collections.emptyList());
            List<Long> l2 = levels2.getOrDefault(i, Collections.emptyList());
            total += (weights[i] / norm) * engine.levelSimilarity(l1, l2, alpha);
        }
        return total;
    }
}
