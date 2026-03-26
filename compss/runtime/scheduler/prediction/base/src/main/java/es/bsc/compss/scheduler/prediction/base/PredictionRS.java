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

import es.bsc.compss.comm.Comm;
import es.bsc.compss.components.impl.ResourceScheduler;
import es.bsc.compss.scheduler.types.AllocatableAction;
import es.bsc.compss.scheduler.types.Score;
import es.bsc.compss.types.TaskDescription;
import es.bsc.compss.types.implementations.Implementation;
import es.bsc.compss.types.resources.Worker;
import es.bsc.compss.types.resources.WorkerResourceDescription;

import org.json.JSONObject;


/**
 * Resource-level scheduler for the prediction policy. Replicates the order-strict resource and implementation scoring
 * logic. A hint-based resource score boost will be added here once the {@link SuccessorHint} rank mechanism and the
 * {@link PredictionSchedulingInformation} hint-transfer flow in {@link PredictionTS} are fully implemented. Score
 * dimensions: - priority — forwarded from the action score. - groupId — action group priority. - resourceScore — 0 by
 * default; +1 for the app-host node. - waitingScore — negative of the number of blocked actions on this worker. -
 * implScore — negative average execution time of the chosen implementation.
 */
public class PredictionRS<T extends WorkerResourceDescription> extends ResourceScheduler<T> {

    /**
     * Constructs a new PredictionRS.
     *
     * @param w Associated worker.
     * @param resJSON JSON description of the worker's resource profile.
     * @param implJSON JSON description of the implementations' profiles.
     */
    public PredictionRS(Worker<T> w, JSONObject resJSON, JSONObject implJSON) {
        super(w, resJSON, implJSON);
    }

    /**
     * Computes the resource-level score for scheduling {@code action} on this worker. Priority and group ID are
     * forwarded from {@code actionScore}. The waiting score is the negative count of blocked actions on this worker.
     * The resource score starts at zero with a +1 bonus if this worker is the application host. TODO: once
     * {@link PredictionTS} fully implements hint transfer via {@link PredictionSchedulingInformation}, add a
     * confidence-weighted boost here for workers that appear in the action's hinted-resource map.
     *
     * @param action Action to be scored.
     * @param params Task description (not used in this implementation).
     * @param actionScore Score produced by the task-level scoring step.
     * @return Resource-level score for this worker.
     */
    @Override
    public Score generateResourceScore(AllocatableAction action, TaskDescription params, Score actionScore) {
        long priority = actionScore.getPriority();
        long groupId = action.getGroupPriority();
        long waitingScore = -this.blocked.size();
        long resourceScore = 0L;

        if (this.myWorker == Comm.getAppHost()) {
            resourceScore++;
        }

        return new Score(priority, groupId, resourceScore, waitingScore, 0);
    }

    /**
     * Computes the implementation-level score for running {@code impl} on this worker. Returns {@code null} if the
     * worker cannot currently satisfy the resource requirements of the implementation, signalling that this (worker,
     * implementation) pair is infeasible. The implementation score is the negative average execution time, so that
     * implementations with shorter observed runtimes receive higher scores. TODO: once hint-transfer is implemented,
     * add a confidence-weighted boost for implementations that appear in the action's hinted-implementation map.
     *
     * @param action Action to be scored.
     * @param params Task description.
     * @param impl Candidate implementation.
     * @param resourceScore Resource-level score already computed for this worker.
     * @return Implementation-level score, or {@code null} if infeasible.
     */
    @Override
    public Score generateImplementationScore(AllocatableAction action, TaskDescription params, Implementation impl,
        Score resourceScore) {
        long priority = resourceScore.getPriority();
        long groupId = action.getGroupPriority();
        long resource = resourceScore.getResourceScore();

        if (!this.myWorker.canRunNow((T) impl.getRequirements())) {
            return null;
        }

        long waitingScore = resourceScore.getWaitingScore();
        long implScore = -this.getProfile(impl).getAverageExecutionTime();

        return new Score(priority, groupId, resource, waitingScore, implScore);
    }
}
