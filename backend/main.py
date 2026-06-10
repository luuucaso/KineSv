from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import tensorflow as tf
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model/modelo_lsu.h5")

with open("model/labels.json", "r") as f:
    labels = json.load(f)

FRAMES_TO_RECORD = 30
TOTAL_FEATURES = 144


class PredictionInput(BaseModel):
    frames: list


@app.get("/")
def root():
    return {"message": "KineSign backend funcionando"}


@app.post("/predict")
def predict(data: PredictionInput):
    frames = np.array(data.frames, dtype=np.float32)

    frames = frames[:FRAMES_TO_RECORD]

    while len(frames) < FRAMES_TO_RECORD:
        empty_frame = np.zeros((TOTAL_FEATURES,), dtype=np.float32)
        frames = np.vstack([frames, empty_frame])

    if frames.shape[1] != TOTAL_FEATURES:
        return {
            "error": f"Formato incorrecto. Esperado {TOTAL_FEATURES}, recibido {frames.shape[1]}"
        }

    X = np.expand_dims(frames, axis=0)

    prediction = model.predict(X, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_label = labels[predicted_index]
    confidence = float(prediction[0][predicted_index])

    return {
        "prediction": predicted_label,
        "confidence": confidence
    }