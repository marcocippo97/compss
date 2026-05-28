package es.bsc.compss.scheduler.rank.base;

import es.bsc.compss.components.impl.TaskScheduler;
import es.bsc.compss.scheduler.exceptions.BlockedActionException;
import es.bsc.compss.scheduler.exceptions.UnassignedActionException;
import es.bsc.compss.scheduler.rank.base.RankBaseTS;
import es.bsc.compss.scheduler.types.AllocatableAction;
import es.bsc.compss.scheduler.types.Score;
import es.bsc.compss.types.allocatableactions.ExecutionAction;
import es.bsc.compss.util.SchedulingOptimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class RankSchedulingOptimizer extends SchedulingOptimizer<RankBaseTS> {

    private static final long DEFER_TIMEOUT_MS = 100L;
    private boolean stop = false;


    public RankSchedulingOptimizer(RankBaseTS ts) {
        super(ts);
    }

    @Override
    public void run() {
        while (!this.stop) {
            if ((this.scheduler.deferredActions != null) && (this.scheduler.deferredDirty)) {
                long now = System.currentTimeMillis();

                // Collect timed-out actions and sort by rank ascending before releasing.
                List<Map.Entry<AllocatableAction, Long>> timedOut = new ArrayList<>();
                for (Map.Entry<AllocatableAction, Long> entry : this.scheduler.deferredActions.entrySet()) {
                    if (now - entry.getValue() >= DEFER_TIMEOUT_MS) {
                        timedOut.add(entry);
                    }
                }
                timedOut.sort((a, b) -> Integer.compare(getRank(a.getKey()), getRank(b.getKey())));

                for (Map.Entry<AllocatableAction, Long> entry : timedOut) {
                    AllocatableAction action = entry.getKey();
                    TaskScheduler.LOGGER
                        .debug(this.scheduler.getLoggerPrefix() + " Deferred action released by timeout: " + action);
                    Score actionScore = scheduler.generateActionScore(action);
                    try {
                        this.scheduler.scheduleAndLaunchAction(action, actionScore);
                        this.scheduler.deferredActions.remove(action);
                    } catch (BlockedActionException | UnassignedActionException e) {
                        TaskScheduler.LOGGER.error(
                            this.scheduler.getLoggerPrefix() + " Error scheduling deferred action: " + action, e);
                    }
                }
            }
            try {
                Thread.sleep(5);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    private int getRank(AllocatableAction action) {
        if (action instanceof ExecutionAction) {
            return ((ExecutionAction) action).getTask().getRank();
        }
        return 0;
    }

    @Override
    public void shutdown() {
        this.stop = true;
        this.interrupt();
    }
}