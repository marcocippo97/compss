package es.bsc.compss.scheduler.rank.prediction.types;

import es.bsc.compss.log.Loggers;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.json.JSONObject;
import org.json.JSONTokener;


public class PredictionSchedulingData {

    private static final Logger LOGGER = LogManager.getLogger(Loggers.TS_COMP);
    private static final String LOG_PREFIX = "[PredictionSchedulingData] ";

    private JSONObject config;

    private Map<Integer, Map<String, ?>> implementations;


    /**
     * Constructs a new {@code PredictionSchedulingData} object.
     */
    public PredictionSchedulingData(String jsonPath) {
        loadFromJson(jsonPath);
    }

    private void loadFromJson(String filename) {
        try (InputStream is = new FileInputStream(filename)) {
            JSONTokener tokener = new JSONTokener(is);
            this.config = new JSONObject(tokener);
            this.implementations = new HashMap<>();

            if (this.config.has("implementations")) {
                JSONObject impls = this.config.getJSONObject("implementations");
                for (String key : impls.keySet()) {
                    try {
                        int id = Integer.parseInt(key);
                        Object val = impls.get(key);
                        if (val instanceof JSONObject) {
                            this.implementations.put(id, ((JSONObject) val).toMap());
                        } else {
                            LOGGER.warn(LOG_PREFIX + " Ignoring malformed implementation entry for rank " + key);
                        }
                    } catch (NumberFormatException nfe) {
                        LOGGER.warn(LOG_PREFIX + " Ignoring non-integer rank key: " + key);
                    }
                }
            } else {
                LOGGER.info(
                    LOG_PREFIX + " No 'implementations' section found: no prioritized implementation will be assigned");
            }

            LOGGER.info(LOG_PREFIX + " Loaded scheduling configuration file at " + filename);
        } catch (Exception e) {
            LOGGER.error(LOG_PREFIX + " Error loading scheduling configuration file at " + filename
                + ", default values will be used: ", e);
            this.implementations = new HashMap<>();
            this.config = new JSONObject();
        }
    }

    /**
     * Get the scheduling implementations.
     */
    public Map<Integer, Map<String, ?>> getImplementations() {
        return this.implementations;
    }

    /**
     * Get the scheduling parameters.
     */
    public Map<String, ?> getParameters() {
        Map<String, ?> parameters;
        try {
            if (this.config != null && this.config.has("parameters")) {
                parameters = this.config.getJSONObject("parameters").toMap();
            } else {
                parameters = new HashMap<>();
            }
        } catch (Exception e) {
            LOGGER.warn(LOG_PREFIX + " No parameters in config file: using default values");
            parameters = new HashMap<>();
        }
        return parameters;
    }
}