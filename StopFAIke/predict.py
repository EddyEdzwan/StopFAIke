import os
from math import sqrt

import tensorflow as tf
import tensorflow_text as text

from pathlib import Path

import pandas as pd
from google.cloud import storage

from sklearn.metrics import mean_absolute_error, mean_squared_error

from StopFAIke.params import BUCKET_NAME, STORAGE_LOCATION, BERT_MODEL_NAME


# root_dir = os.path.dirname(os.path.dirname(__file__))
# LOCAL_PATH = os.path.join(root_dir, "raw_data", "test.csv")

# GCP version
# def get_test_data(local=False):
#     """method to get the training data (or a portion of it) from google cloud bucket
#     To predict we can either obtain predictions from train data or from test data"""
#     # Add Client() here
#     if local:
#         path = LOCAL_PATH
#     else:
#         path = f"gs://{BUCKET_NAME}/{BUCKET_TEST_DATA_PATH}"
#     df = pd.read_csv(path)
#     return df


def get_model_from_gcp():
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=STORAGE_LOCATION)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(blob.name)

    # Load the model downloaded locally from CGP
    reloaded_model = tf.saved_model.load(STORAGE_LOCATION)
    return reloaded_model


# def evaluate_model(y, y_pred):
#     MAE = round(mean_absolute_error(y, y_pred), 2)
#     RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
#     res = {'MAE': MAE, 'RMSE': RMSE}
#     return res


# def generate_submission_csv(kaggle_upload=False):

#     # Get data from GCP
#     df_test = get_test_data(local=False)
#     print("shape: {}".format(df_test.shape))
#     print("size: {} Mb".format(df_test.memory_usage().sum() / 1e6))

#     # Get data from GCP model
#     # pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
#     pipeline = get_model(local=False)

#     if "best_estimator_" in dir(pipeline):
#         y_pred = pipeline.best_estimator_.predict(df_test)
#     else:
#         y_pred = pipeline.predict(df_test)

#     # Create Kaggle csv file submission
#     df_test["fare_amount"] = y_pred
#     df_sample = df_test[["key", "fare_amount"]]
#     name = f"predictions_test_ex.csv"
#     df_sample.to_csv(name, index=False)
#     print("prediction saved under kaggle format")

#     # Set kaggle_upload to False unless you install kaggle cli
#     if kaggle_upload:
#         kaggle_message_submission = name[:-4]
#         command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
#         os.system(command)


if __name__ == '__main__':

    print("#### Loading the model ...")
    reloaded_model = get_model_from_gcp()
    print("#### Model uploaded ...")

    samples = ['Julien is 43yo',
           'Nina and Fleur are the daughters of Julien',
           'Trump president']

    for sample in samples:
        y_prob = reloaded_model([sample])
        print(f"Pred: {y_prob.numpy()[0][0]:.3f} - {sample} ")
        # model = get_model_from_gcp()
