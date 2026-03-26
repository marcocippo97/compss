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
import es.bsc.compss.scheduler.types.SchedulingInformation;
import es.bsc.compss.types.implementations.Implementation;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;


/**
 * Extension of {@link SchedulingInformation} that carries resource and implementation hints produced by the
 * similarity-based look-ahead mechanism. An instance of this class is associated with every action created by
 * {@code PredictionTS.generateSchedulingInformation}. When a similarity match causes hints to be written (in
 * {@code PredictionTS.handleDependencyFreeActions}), they are stored here and later read by
 * {@code PredictionRS.generateResourceScore} to boost the score of the suggested resource. When multiple past tasks
 * independently suggest the same resource for a given future task, the stored confidence is the maximum observed value,
 * preventing score inflation from repeated confirmations.
 */
public class PredictionSchedulingInformation extends SchedulingInformation {

    /**
     * Map from a hinted ResourceScheduler to the maximum confidence accumulated across all hints that pointed to it.
     */
    private final Map<ResourceScheduler<?>, Double> hintedResources = new HashMap<>();

    /**
     * Optional per-implementation hints for finer-grained suggestions (e.g. preferring a GPU variant over a CPU variant
     * of the same coreId).
     */
    private final Map<Implementation, Double> hintedImplementations = new HashMap<>();


    /**
     * Constructs a PredictionSchedulingInformation for the given resource.
     *
     * @param rs ResourceScheduler initially assigned to the action, or {@code null} if not yet assigned.
     */
    public PredictionSchedulingInformation(ResourceScheduler<?> rs) {
        super(rs);
    }

    /**
     * Records or strengthens a resource hint. If the resource has already been hinted, the higher of the existing and
     * new confidence values is retained.
     *
     * @param resource Hinted ResourceScheduler.
     * @param confidence Confidence weight in [0.0, 1.0].
     */
    public void addHintedResource(ResourceScheduler<?> resource, double confidence) {
        hintedResources.merge(resource, confidence, Math::max);
    }

    /**
     * Returns an unmodifiable view of the hinted-resource map.
     *
     * @return Map from ResourceScheduler to maximum confidence; never null.
     */
    public Map<ResourceScheduler<?>, Double> getHintedResources() {
        return Collections.unmodifiableMap(hintedResources);
    }

    /**
     * Records or strengthens an implementation hint. If the implementation has already been hinted, the higher
     * confidence is retained.
     *
     * @param impl Hinted Implementation.
     * @param confidence Confidence weight in [0.0, 1.0].
     */
    public void addHintedImplementation(Implementation impl, double confidence) {
        hintedImplementations.merge(impl, confidence, Math::max);
    }

    /**
     * Returns whether an implementation hint exists for the given implementation.
     *
     * @param impl Implementation to query.
     * @return {@code true} if a hint is present.
     */
    public boolean hasHintedImplementation(Implementation impl) {
        return hintedImplementations.containsKey(impl);
    }

    /**
     * Returns the confidence of an implementation hint, or 0.0 if none exists.
     *
     * @param impl Implementation to query.
     * @return Confidence in [0.0, 1.0].
     */
    public double getHintedImplementationConfidence(Implementation impl) {
        return hintedImplementations.getOrDefault(impl, 0.0);
    }
}
