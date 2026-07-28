"""
Crop Recommendation API
------------------------
Wraps the trained RandomForest model + LabelEncoder into a REST API
so it can be called from a website, mobile app, or another service.

Run locally with:
    uvicorn app:app --reload

Then open: http://127.0.0.1:8000/docs
(FastAPI auto-generates an interactive test page there)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# -----------------------------
# 1. Load the trained model + encoder ONCE when the server starts
#    (loading on every request would be slow)
# -----------------------------
try:
    model = joblib.load("Models/Random_forest_model_Intelligence.pkl")
    encoder = joblib.load("Models/label_Encoded_Intelligence.pkl")
except FileNotFoundError:
    model = None
    encoder = None

app = FastAPI(
    title="Crop Recommendation API",
    description="Predicts the most suitable crop based on soil NPK values, humidity, and rainfall.",
    version="1.0.0"
)


# -----------------------------
# 2. Define the input schema
#    This tells FastAPI (and anyone calling the API) exactly what
#    fields are required, their types, and validation rules.
# -----------------------------
class CropInput(BaseModel):
    N: float = Field(..., description="Nitrogen content in soil", example=90)
    P: float = Field(..., description="Phosphorus content in soil", example=42)
    K: float = Field(..., description="Potassium content in soil", example=43)
    humidity: float = Field(..., description="Relative humidity in %", example=82.0)
    rainfall: float = Field(..., description="Rainfall in mm", example=202.9)


class CropOutput(BaseModel):
    recommended_crop: str
    confidence: float


# -----------------------------
# 3. Health check endpoint
#    Useful so you (or a monitoring tool) can check the API is alive
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Crop Recommendation API is running"}


# -----------------------------
# 4. The main prediction endpoint
# -----------------------------
@app.post("/predict", response_model=CropOutput)
def predict_crop(data: CropInput):
    if model is None or encoder is None:
        raise HTTPException(
            status_code=500,
            detail="Model files not found. Place 'Random_forest_model_Intelligence.pkl' "
                   "and 'label_Encoded_Intelligence.pkl' in this folder."
        )

    # Convert input into the 2D array shape the model expects
    features = np.array([[data.N, data.P, data.K, data.humidity, data.rainfall]])

    # Run prediction
    prediction_encoded = model.predict(features)
    prediction_label = encoder.inverse_transform(prediction_encoded)[0]

    # Get confidence score (probability of the predicted class)
    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))

    return CropOutput(recommended_crop=prediction_label, confidence=round(confidence, 4))