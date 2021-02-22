import tensorflow as tf
import os
import numpy as np
import logging
import json
from pathlib import Path


LOGGER = logging.getLogger(__name__)
LOGGER.warn("entered entry script")


def init():
    global model
    LOGGER.warn("initializing...")
    model_path = Path(os.getenv('AZUREML_MODEL_DIR')) / 'baseline_cnn_mitbih/1'
    model = tf.keras.models.load_model(model_path)
    LOGGER.warn("model loaded")
    
    
def run(json_data):
    LOGGER.warn("task received")
    try:
#         sequence = np.array(json.loads(json_data))
#         windows = cut_to_windows(sequence)  # -> shape (x, 187, 1)      
#         result = model.predict(windows)
        data = np.array(json.loads(json_data)
        result = model.predict(data)
    
        LOGGER.warn("done")
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        LOGGER.error(error)
        return error
