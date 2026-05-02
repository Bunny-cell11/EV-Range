import joblib
import numpy as np
from src.config import MODEL_PATH, SCALER_PATH

def predict_range(input_data):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)

    return model.predict(input_scaled)[0]
