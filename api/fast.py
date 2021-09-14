import numpy as np

import tensorflow as tf
import tensorflow_text as text

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from StopFAIke.predict import get_model_from_gcp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get the model (locally or from GCP)
reloaded_model = get_model_from_gcp(local=True)


@app.get("/")
def index():
    return {'message': 'This is StopFAIke API!'}


@app.get("/predict")
def predict(article):

    X = dict(article=[article])

    y_prob = reloaded_model(X['article'])

    print(f"Probability (0 (True) - 1 (Fake)): {np.round(y_prob.numpy()[0][0], 3)}")

    return {'prediction': float(y_prob.numpy()[0][0])}
