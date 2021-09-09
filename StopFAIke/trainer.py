
from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import joblib

### GCP Project - - - - - - - - - - - - - - - - - - - - - -
PROJECT_ID = 'le-wagon-data-bootcamp-321006'

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -
BUCKET_NAME = 'wagon-data-615-seguy'
BUCKET_FOLDER = 'data'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
P_TRAIN_DATA_PATH = 'data/politifact_scrap.csv'
FNN_TRAIN_DATA_PATH = 'data/FakesNewsNET.csv'
BIS_T_BUCKET_TRAIN_DATA_PATH = 'data/True.csv'
BIS_F_BUCKET_TRAIN_DATA_PATH = 'data/Fake.csv'
PO_BUCKET_TRAIN_DATA_PATH = 'data/poynter_final_condensed.csv'


# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'StopFAIke'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'


def get_data_from_gcp(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, nrows=20000):
    """method to get the training data (or a portion of it) from GCP"""
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=nrows)
    return df


# def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
#     """method to get the training data (or a portion of it) from GCP"""
#     path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/politifact_scrap.csv'
#     df = pd.read_csv(path, nrows=nrows)
#     return df


# def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
#     """method to get the training data (or a portion of it) from GCP"""
#     path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/FakesNewsNET.csv'
#     df = pd.read_csv(path, nrows=nrows)
#     return df


# def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
#     """method to get the training data (or a portion of it) from GCP"""
#     true_path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/True.csv'
#     fake_path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/Fake.csv'
#     true_df = pd.read_csv(true_path, nrows=nrows)
#     fake_df = pd.read_csv(fake_path, nrows=nrows)
#     return true_df, fake_df


# def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
#     """method to get the training data (or a portion of it) from GCP"""
#     path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/poynter_final_condensed.csv'
#     df = pd.read_csv(path, nrows=nrows)
#     return df



def preprocess(df):
    """method that pre-process the data"""
    df["distance"] = compute_distance(df)
    X_train = df[["distance"]]
    y_train = df["fare_amount"]
    return X_train, y_train


def train_model(X_train, y_train):
    """method that trains the model"""
    rgs = linear_model.Lasso(alpha=0.1)
    rgs.fit(X_train, y_train)
    print("trained model")
    return rgs


STORAGE_LOCATION = 'models/simpletaxifare/model.joblib'


def upload_model_to_gcp():


    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == '__main__':
    # get training data from GCP bucket
    F_df = get_data_from_gcp(BUCKET_NAME, P_TRAIN_DATA_PATH)
    F_df = get_data_from_gcp(BUCKET_NAME, P_TRAIN_DATA_PATH)
    F_df = get_data_from_gcp(BUCKET_NAME, P_TRAIN_DATA_PATH)
    F_df = get_data_from_gcp(BUCKET_NAME, P_TRAIN_DATA_PATH)
    F_df = get_data_from_gcp(BUCKET_NAME, P_TRAIN_DATA_PATH)




    # preprocess data
    X_train, y_train = preprocess(df)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    reg = train_model(X_train, y_train)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(reg)
