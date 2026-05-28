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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


/**
 * Stateful engine that implements the multi-level, weighted similarity metric used by {@link WLSimilarityFunction}. The
 * engine is updated incrementally as new tasks arrive: each call to {@link #updateFeatureSpace} registers the numerical
 * feature keys of a new node so that cross-node vector projections remain consistent. All intermediate similarity
 * computations operate directly on task IDs, resolving full {@link TaskGraphCache.NodeInfo} objects from the shared
 * cache. Thread safety: methods that mutate {@code featureIndex} and {@code numericStats} are {@code synchronized};
 * read-only query methods are not and should be called from a single scheduler thread.
 */
public class SimilarityEngine {

    /**
     * Maps the qualified key {@code "taskName::paramKey"} to its assigned column index in the shared feature vector.
     * Ensures that vectors built for different nodes use identical dimension ordering.
     */
    private final Map<String, Integer> featureIndex = new HashMap<>();
    private int nextFeatureIdx = 0;

    /**
     * Per-key running min/max statistics stored as {@code double[]{min, max}}. Maintained for potential future global
     * normalisation; not applied during similarity computation because the cosine-x-magnitude metric is scale-invariant
     * and normalisation would distort comparisons when per-key sample counts are small.
     */
    private final Map<String, double[]> numericStats = new HashMap<>();

    private final TaskGraphCache cache;


    /**
     * Constructs a SimilarityEngine backed by the given cache.
     *
     * @param cache Shared task-graph cache.
     */
    public SimilarityEngine(TaskGraphCache cache) {
        this.cache = cache;
    }

    // -------------------------------------------------------------------------
    // Feature-space maintenance
    // -------------------------------------------------------------------------

    /**
     * Registers the numerical features of a newly added task node in the feature index and updates running min/max
     * statistics. Must be called once per new task, after {@link TaskGraphCache#registerTask}, and before any
     * similarity query that involves the new node.
     *
     * @param taskId Identifier of the newly registered task.
     */
    public synchronized void updateFeatureSpace(long taskId) {
        TaskGraphCache.NodeInfo node = cache.getNode(taskId);
        if (node == null) {
            return;
        }
        String taskName = node.features.getTaskName();

        for (Map.Entry<String, double[]> e : node.features.getNumericalFeatures().entrySet()) {
            String key = taskName + "::" + e.getKey();
            double[] values = e.getValue();

            featureIndex.putIfAbsent(key, nextFeatureIdx++);

            double[] stats = numericStats.computeIfAbsent(key, k -> new double[] { Double.MAX_VALUE,
                -Double.MAX_VALUE });
            for (double v : values) {
                if (v < stats[0]) {
                    stats[0] = v;
                }
                if (v > stats[1]) {
                    stats[1] = v;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Numeric vector projection
    // -------------------------------------------------------------------------

    /**
     * Builds two raw numeric vectors restricted to feature keys present in both nodes. Dimensions are ordered
     * lexicographically by qualified key to ensure stable, reproducible alignment across calls. No global normalisation
     * is applied: the cosine-x-magnitude metric is scale-invariant, and normalisation would distort comparisons when
     * per-key sample sizes are small. Each component is the arithmetic mean of the raw values stored under the
     * corresponding key. Returns a pair {@code double[2][]} where {@code [0]} and {@code [1]} are the vectors for
     * {@code id1} and {@code id2} respectively. Both arrays are empty if the nodes share no numerical feature keys.
     *
     * @param id1 First task ID.
     * @param id2 Second task ID.
     * @return A {@code double[2][]} pair of equal-length arrays.
     */
    public double[][] numericVectorCommon(long id1, long id2) {
        TaskGraphCache.NodeInfo n1 = cache.getNode(id1);
        TaskGraphCache.NodeInfo n2 = cache.getNode(id2);
        if (n1 == null || n2 == null) {
            return new double[][] { new double[0],
                new double[0] };
        }

        String t1 = n1.features.getTaskName();
        String t2 = n2.features.getTaskName();

        Set<String> keys1 = new HashSet<>();
        Set<String> keys2 = new HashSet<>();
        for (String k : n1.features.getNumericalFeatures().keySet()) {
            keys1.add(t1 + "::" + k);
        }
        for (String k : n2.features.getNumericalFeatures().keySet()) {
            keys2.add(t2 + "::" + k);
        }

        List<String> common = new ArrayList<>(keys1);
        common.retainAll(keys2);
        Collections.sort(common);

        if (common.isEmpty()) {
            return new double[][] { new double[0],
                new double[0] };
        }

        double[] vec1 = new double[common.size()];
        double[] vec2 = new double[common.size()];
        for (int i = 0; i < common.size(); i++) {
            String param = common.get(i).split("::", 2)[1];
            vec1[i] = mean(n1.features.flattenNumeric(param));
            vec2[i] = mean(n2.features.flattenNumeric(param));
        }
        return new double[][] { vec1,
            vec2 };
    }

    // -------------------------------------------------------------------------
    // Similarity primitives
    // -------------------------------------------------------------------------

    /**
     * Computes the Jaccard similarity between two sets of strings. Returns 1.0 when both sets are empty (vacuous
     * equality). Returns 0.0 when exactly one set is empty (maximum dissimilarity).
     *
     * @param s1 First set.
     * @param s2 Second set.
     * @return Jaccard similarity in [0.0, 1.0].
     */
    public static double jaccardSimilarity(Set<String> s1, Set<String> s2) {
        if (s1.isEmpty() && s2.isEmpty()) {
            return 1.0;
        }
        Set<String> union = new HashSet<>(s1);
        union.addAll(s2);
        if (union.isEmpty()) {
            return 0.0;
        }
        Set<String> inter = new HashSet<>(s1);
        inter.retainAll(s2);
        return (double) inter.size() / union.size();
    }

    /**
     * Computes the cosine-times-magnitude-ratio similarity between two vectors. The result combines angular similarity
     * (cosine) with scale similarity (magnitude ratio), so that vectors pointing in the same direction but with very
     * different norms receive a low score: result = cosine(v1, v2) * min(|v1|, |v2|) / max(|v1|, |v2|). Special cases:
     * both norms zero returns 1.0 (identical zero vectors); exactly one zero returns 0.0 (maximally dissimilar).
     *
     * @param v1 First vector.
     * @param v2 Second vector (must have the same length as v1).
     * @return Similarity value in [0.0, 1.0].
     */
    public static double cosineXMagnitude(double[] v1, double[] v2) {
        double n1 = norm(v1);
        double n2 = norm(v2);
        if (n1 == 0.0 && n2 == 0.0) {
            return 1.0;
        }
        if (n1 == 0.0 || n2 == 0.0) {
            return 0.0;
        }
        double cosSim = dot(v1, v2) / (n1 * n2);
        double magRatio = Math.min(n1, n2) / Math.max(n1, n2);
        return cosSim * magRatio;
    }

    // -------------------------------------------------------------------------
    // Optimal numeric matching (Hungarian algorithm)
    // -------------------------------------------------------------------------

    /**
     * Computes the optimal average pairwise numeric similarity across two lists of nodes that share the same
     * categorical hash, using the Hungarian (minimum-cost bipartite matching) algorithm. A cost matrix is built where
     * {@code cost[i][j] = 1 - sim(ids1[i], ids2[j])} and similarity is measured by {@link #cosineXMagnitude} applied to
     * the common-key numeric vectors. Identical node IDs receive similarity 1.0 (cost 0). Node pairs with no common
     * numerical features receive similarity 0.0 (cost 1). The return value is the average similarity over all matched
     * pairs. Complexity: O(n³) in the number of nodes per bucket. This is acceptable for the expected small
     * cardinalities of each categorical-hash bucket. If buckets grow large, a greedy approximation should be
     * considered.
     *
     * @param ids1 Task IDs from the first node set (same categorical hash).
     * @param ids2 Task IDs from the second node set (same categorical hash).
     * @return Average similarity of the optimal matching in [0.0, 1.0].
     */
    public double optimalNumericMatching(List<Long> ids1, List<Long> ids2) {
        int n = ids1.size();
        int m = ids2.size();
        if (n == 0 || m == 0) {
            return 0.0;
        }

        double[][] cost = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (ids1.get(i).equals(ids2.get(j))) {
                    cost[i][j] = 0.0;
                } else {
                    double[][] vecs = numericVectorCommon(ids1.get(i), ids2.get(j));
                    double sim = (vecs[0].length > 0) ? cosineXMagnitude(vecs[0], vecs[1]) : 0.0;
                    cost[i][j] = 1.0 - sim;
                }
            }
        }

        int[][] assignment = hungarian(cost);
        double total = 0.0;
        for (int[] pair : assignment) {
            total += 1.0 - cost[pair[0]][pair[1]];
        }
        return (assignment.length > 0) ? total / assignment.length : 0.0;
    }

    // -------------------------------------------------------------------------
    // Level similarity
    // -------------------------------------------------------------------------

    /**
     * Computes the similarity between two sets of nodes at the same BFS level of the DAG using a two-step structural
     * and numerical comparison. Algorithm: - Group nodes in each set by their categorical hash. - Compute Jaccard
     * similarity on the two sets of hash keys. - For each hash key present in both sets, run
     * {@link #optimalNumericMatching}. - numericScore = arithmetic mean of per-bucket scores across all common hashes.
     * - Return jaccardScore * numericScore^(1 + alpha). The {@code alpha} parameter controls how strongly numerical
     * dissimilarity penalises the score: alpha = 0.0 gives a linear penalty (jaccardScore * numericScore); alpha = 1.0
     * gives a quadratic penalty. Returns 1.0 when both lists are empty (vacuous equality).
     *
     * @param ids1 Task IDs from the first node set at this level.
     * @param ids2 Task IDs from the second node set at this level.
     * @param alpha Non-negative exponent that shapes the numerical penalty.
     * @return Level similarity score in [0.0, 1.0].
     */
    public double levelSimilarity(List<Long> ids1, List<Long> ids2, double alpha) {
        if (ids1.isEmpty() && ids2.isEmpty()) {
            return 1.0;
        }

        Map<String, List<Long>> map1 = groupByCategoricalHash(ids1);
        Map<String, List<Long>> map2 = groupByCategoricalHash(ids2);

        double jaccardScore = jaccardSimilarity(map1.keySet(), map2.keySet());

        Set<String> commonHashes = new HashSet<>(map1.keySet());
        commonHashes.retainAll(map2.keySet());

        double sumF = 0.0;
        for (String hash : commonHashes) {
            sumF += optimalNumericMatching(map1.get(hash), map2.get(hash));
        }
        double numericScore = commonHashes.isEmpty() ? 0.0 : sumF / commonHashes.size();

        return jaccardScore * Math.pow(numericScore, 1.0 + alpha);
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private Map<String, List<Long>> groupByCategoricalHash(List<Long> ids) {
        Map<String, List<Long>> map = new HashMap<>();
        for (Long id : ids) {
            TaskGraphCache.NodeInfo node = cache.getNode(id);
            if (node != null) {
                map.computeIfAbsent(node.features.categoricalHash(), k -> new ArrayList<>()).add(id);
            }
        }
        return map;
    }

    private static double mean(double[] values) {
        if (values.length == 0) {
            return 0.0;
        }
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    private static double norm(double[] v) {
        double sum = 0.0;
        for (double x : v) {
            sum += x * x;
        }
        return Math.sqrt(sum);
    }

    private static double dot(double[] v1, double[] v2) {
        double sum = 0.0;
        for (int i = 0; i < v1.length; i++) {
            sum += v1[i] * v2[i];
        }
        return sum;
    }

    /**
     * Solves the linear sum assignment problem on the given cost matrix using the Kuhn-Munkres (Hungarian) algorithm.
     * Rectangular matrices are handled by padding the shorter dimension with a large constant so that the padded square
     * matrix is always n x n where n = max(rows, cols). Padding cells are never part of the returned assignment. The
     * algorithm runs in O(n³) time and O(n²) space. Implementation follows the standard potential-based formulation:
     * u[i] and v[j] are row and column potentials maintained so that u[i] + v[j] <= cost[i][j] for all (i,j)
     * (feasibility). Augmenting paths are found via a shortest-path scan (Dijkstra-like) over the reduced cost matrix
     * cost[i][j] - u[i] - v[j].
     *
     * @param cost n x m cost matrix; must be non-null and non-empty.
     * @return Array of min(n, m) matched {row, col} index pairs, all within the original dimensions.
     */
    private static int[][] hungarian(double[][] cost) {
        int rows = cost.length;
        int cols = cost[0].length;
        // Work on a square matrix of size dim x dim by virtual padding.
        int dim = Math.max(rows, cols);
        final double INF = Double.MAX_VALUE / 2.0;
        // Pad value used for dummy cells; must be larger than any real cost but not INF to avoid
        // overflow during arithmetic.
        final double PAD = 1.0e15;

        // u[i] = row potential for row i (1-indexed; u[0] unused).
        // v[j] = col potential for col j (1-indexed; v[0] is the dummy row potential).
        double[] u = new double[dim + 1];
        double[] v = new double[dim + 1];
        // match[j] = row currently matched to column j (0 = unmatched).
        int[] matchCol = new int[dim + 1];
        // matchRow[i] = column currently matched to row i (0 = unmatched).
        int[] matchRow = new int[dim + 1];

        for (int i = 1; i <= dim; i++) {
            // Extend the assignment by one row at a time using the shortest augmenting path.
            // way[j] = the column that led to column j along the current shortest path.
            int[] way = new int[dim + 1];
            double[] minDist = new double[dim + 1];
            boolean[] used = new boolean[dim + 1];

            for (int j = 0; j <= dim; j++) {
                minDist[j] = INF;
            }
            matchCol[0] = i;
            int curCol = 0;

            do {
                used[curCol] = true;
                int nextCol = -1;
                double delta = INF;
                int curRow = matchCol[curCol];

                for (int j = 1; j <= dim; j++) {
                    if (used[j]) {
                        continue;
                    }
                    // Cost of the current row (padded if outside original dimensions).
                    double c = (curRow <= rows && j <= cols) ? cost[curRow - 1][j - 1] : PAD;
                    double val = c - u[curRow] - v[j];
                    if (val < minDist[j]) {
                        minDist[j] = val;
                        way[j] = curCol;
                    }
                    if (minDist[j] < delta) {
                        delta = minDist[j];
                        nextCol = j;
                    }
                }

                // Update potentials by delta.
                for (int j = 0; j <= dim; j++) {
                    if (used[j]) {
                        u[matchCol[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minDist[j] -= delta;
                    }
                }

                curCol = nextCol;
            } while (matchCol[curCol] != 0);

            // Augment along the path recorded in way[].
            do {
                int prevCol = way[curCol];
                matchCol[curCol] = matchCol[prevCol];
                matchRow[matchCol[curCol]] = curCol;
                curCol = prevCol;
            } while (curCol != 0);
        }

        // Collect only assignments within the original (non-padded) matrix dimensions.
        List<int[]> result = new ArrayList<>();
        for (int j = 1; j <= dim; j++) {
            int row = matchCol[j];
            if (row >= 1 && row <= rows && j <= cols) {
                result.add(new int[] { row - 1,
                    j - 1 });
            }
        }
        return result.toArray(new int[0][]);
    }
}
