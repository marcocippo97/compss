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
import es.bsc.compss.scheduler.rank.base.RankBaseRS;
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
public class PredictionRS<T extends WorkerResourceDescription> extends RankBaseRS<T> {

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
}