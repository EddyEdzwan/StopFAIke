import numpy as np

import pandas as pd

import shap

import tensorflow as tf
import tensorflow_text as text

from transformers import AutoTokenizer

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from StopFAIke.predict import get_model_from_gcp

from StopFAIke.utils import clean

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

# Initialising the bert tokenizer for use in shap.Explainer
model_name = "bert-base-uncased"
bert_tokenizer_pretrained = AutoTokenizer.from_pretrained(model_name)

#Creating custom function for shap.Explainer
def f(x):
    '''
    Returns the class probabilities output by the custom BERT classifier - two classes  
    '''
    # Calculate the class probabilities and stack them for each input sentence
    prob_class_0, prob_class_1 = 1 - reloaded_model(x), reloaded_model(x)
    stacked_probabilities = np.hstack((prob_class_0, prob_class_1))
    
    # Calculate the logit values, change return value if choosing to explain via logits instead 
    # val = sp.special.logit(stacked_probabilities)
    
    return stacked_probabilities

# Initialising the shap.Explainer object
explainer = shap.Explainer(f, masker=bert_tokenizer_pretrained, output_names=['TRUE', 'FAKE'])

@app.get("/")
def index():
    return {'message': 'This is StopFAIke API!'}


@app.get("/predict")
def predict(article):

    X = dict(article=article)

    print(X)

    # cleaning
    X_clean = clean(X['article'], stopword=False, lemmat=False)

    # prediction
    y_prob = reloaded_model([X_clean])

    print(type(X_clean))

    return {'prediction': float(y_prob.numpy()[0][0])}

@app.get("/shapvalues")
def shapvalues(article):

    X = dict(article=article)

    # cleaning
    X_clean = clean(X['article'], stopword=False, lemmat=False)

    # prediction
    y_prob = reloaded_model([X_clean])

    shap_values = explainer([X_clean])

    if y_prob.numpy()[0][0] >= 0.5:
        return {"values" : list(shap_values.values[0, :, 1]),
        "base_values" : float(shap_values.base_values[0, 1]),
        "data" : list(shap_values.data[0]),
        "output_names" : str(shap_values.output_names[1])}
    
    return {"values" : list(shap_values.values[0, :, 0]),
        "base_values" : float(shap_values.base_values[0, 0]),
        "data" : list(shap_values.data[0]),
        "output_names" : str(shap_values.output_names[0])}