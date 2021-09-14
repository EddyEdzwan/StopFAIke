import numpy as np
import pandas as pd

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
def predict(text):

    X = dict(text=[text])

    # X = pd.DataFrame(dict(text=[text]))

    # Process the input as list
    # X = [text]

    # Get the model (locally or from GCP)
    # reloaded_model = get_model_from_gcp(local=True)

    # Make prediction
    # y_prob = reloaded_model(X['text'].values.tolist())

    y_prob = reloaded_model(X['text'])

    return {'prediction': y_prob}
