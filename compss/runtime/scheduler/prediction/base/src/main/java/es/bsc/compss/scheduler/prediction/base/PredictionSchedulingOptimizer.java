package es.bsc.compss.scheduler.prediction.base;

import es.bsc.compss.components.impl.TaskScheduler;
import es.bsc.compss.scheduler.exceptions.BlockedActionException;
import es.bsc.compss.scheduler.exceptions.UnassignedActionException;
import es.bsc.compss.scheduler.types.AllocatableAction;
import es.bsc.compss.scheduler.types.Score;
import es.bsc.compss.util.SchedulingOptimizer;

import java.util.Iterator;
import java.util.Map;


public class PredictionSchedulingOptimizer extends SchedulingOptimizer<PredictionTS> {

    /**
     * Maximum time in milliseconds that a deferred action waits for a lower-ranked sibling before being submitted
     * unconditionally to prevent starvation.
     */
    private static final long DEFER_TIMEOUT_MS = 100L;
    private boolean stop = false;


    public PredictionSchedulingOptimizer(PredictionTS ts) {
        super(ts);
    }

    @Override
    public void run() {
        while (!this.stop) {

            if (this.scheduler.deferredActions == null) {
                continue;
            }
            Iterator<Map.Entry<AllocatableAction, Long>> deferIter =
                this.scheduler.deferredActions.entrySet().iterator();

            while (deferIter.hasNext()) {
                Map.Entry<AllocatableAction, Long> entry = deferIter.next();
                long now = System.currentTimeMillis();
                if (now - entry.getValue() >= DEFER_TIMEOUT_MS) {
                    AllocatableAction action = entry.getKey();
                    TaskScheduler.LOGGER.debug("[PredictionTS] Deferred action released by timeout: " + action);
                    Score actionScore = scheduler.generateActionScore(action);
                    try {
                        this.scheduler.scheduleAndLaunchAction(action, actionScore);
                        deferIter.remove();
                    } catch (BlockedActionException | UnassignedActionException e) {
                        TaskScheduler.LOGGER.error("[PredictionTS] Error scheduling deferred action: " + action, e);
                    }
                }
            }
            // try {
            // Thread.sleep(DEFER_TIMEOUT_MS / 2);
            // } catch (InterruptedException e) {
            // Thread.currentThread().interrupt();
            // }
        }
    }

    @Override
    public void shutdown() {
        this.stop = true; // TODO is it necessary?
        this.interrupt();
    }
}