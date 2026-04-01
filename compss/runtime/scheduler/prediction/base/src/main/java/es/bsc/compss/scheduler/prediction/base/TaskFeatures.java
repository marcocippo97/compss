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

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;


/**
 * Immutable container for the categorical and numerical features of a single task node in the DAG. Categorical features
 * are extracted from the CoreElement associated with the task; the mandatory key is "task_name". Numerical features
 * represent measurable task parameters stored as double arrays; their source is TBD.
 */
public class TaskFeatures {

    private final int coreElementId;
    private final Map<String, String> categoricalFeatures;
    private final Map<String, double[]> numericalFeatures;


    /**
     * Constructs a TaskFeatures container.
     *
     * @param coreElementId COMPSs core element identifier for this task type.
     * @param categorical Map of categorical feature names to string values. Must contain the key "task_name".
     * @param numerical Map of numerical feature names to raw double arrays.
     */
    public TaskFeatures(int coreElementId, Map<String, String> categorical, Map<String, double[]> numerical) {
        this.categoricalFeatures = Collections.unmodifiableMap(new HashMap<>(categorical));
        this.numericalFeatures = Collections.unmodifiableMap(new HashMap<>(numerical));
        this.coreElementId = coreElementId;
    }

    /**
     * Returns the COMPSs core element identifier for this task type.
     *
     * @return Core element ID.
     */
    public int getCoreId() {
        return coreElementId;
    }

    /**
     * Returns the full map of categorical features.
     *
     * @return Unmodifiable categorical feature map.
     */
    public Map<String, String> getCategoricalFeatures() {
        return categoricalFeatures;
    }

    /**
     * Returns the full map of numerical features.
     *
     * @return Unmodifiable numerical feature map.
     */
    public Map<String, double[]> getNumericalFeatures() {
        return numericalFeatures;
    }

    /**
     * Returns the mandatory "task_name" categorical feature.
     *
     * @return Task name string, or {@code null} if not set.
     */
    public String getTaskName() {
        return categoricalFeatures.get("task_name");
    }

    /**
     * Returns the named numerical feature as a raw double[]. Returns an empty array if the key is absent.
     *
     * @param key Feature name.
     * @return Feature values array, never null.
     */
    public double[] flattenNumeric(String key) {
        double[] v = numericalFeatures.get(key);
        return (v != null) ? v : new double[0];
    }

    /**
     * Computes a stable SHA-256 hex digest of the categorical features sorted by key. Used in
     * {@link SimilarityEngine#levelSimilarity} to group nodes that share the same categorical signature.
     *
     * @return Hex-encoded SHA-256 digest string.
     */
    public String categoricalHash() {
        String sorted = new TreeSet(categoricalFeatures.keySet()).toString();
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(sorted.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(hash.length * 2);
            for (byte b : hash) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            // SHA-256 is guaranteed to be available in all JVMs.
            throw new RuntimeException("SHA-256 not available", e);
        }
    }
}
