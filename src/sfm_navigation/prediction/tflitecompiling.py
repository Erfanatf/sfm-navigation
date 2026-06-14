"""Save a TFLite‑compatible model (force CPU LSTM, no CuDNN)."""
import os
os.environ["TF_ENABLE_CUDNN_RNN"] = "0"        # ← must be set before TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Input

# ------------------------------------------------------------------
# 1. Re‑create the model using CPU‑only LSTM
# ------------------------------------------------------------------
model = Sequential([
    Input(shape=(30, 5)),
    LSTM(128, activation='tanh', return_sequences=True,
         implementation=1),                     # standard TF ops, TFLite ok
    LSTM(128, activation='tanh', return_sequences=True,
         implementation=1),
    TimeDistributed(Dense(3))
])

# ------------------------------------------------------------------
# 2. Transfer weights from the trained CuDNN model
# ------------------------------------------------------------------
trained = load_model(
    "src/sfm_navigation/prediction/trained_model.keras",
    compile=False
)
model.set_weights(trained.get_weights())
print("Weights transferred successfully.")

# ------------------------------------------------------------------
# 3. Convert to TFLite (with optimizations)
# ------------------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open("src/sfm_navigation/prediction/lstm_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved to src/sfm_navigation/prediction/lstm_model.tflite")