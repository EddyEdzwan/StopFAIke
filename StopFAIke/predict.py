
"""Predict with trained model"""

import pandas as pd

import tensorflow as tf
import tensorflow_text as text

from pathlib import Path
from google.cloud import storage

from StopFAIke.params import BUCKET_NAME, STORAGE_LOCATION

from StopFAIke.utils import clean

# root_dir = os.path.dirname(os.path.dirname(__file__))
# LOCAL_PATH = os.path.join(root_dir, "raw_data", "test.csv")


def get_model_from_gcp(local=False):
    """
    Download model from GCP Storage

    Args:
        local: if False, model is loaded locally. If True, downloaded from GCP Storage first and loaded locally.

    Returns:
        Tensorflow model
    """

    if local:
        # Load the model locally
        reloaded_model = tf.saved_model.load(STORAGE_LOCATION)
    else:
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

        # Load the model downloaded from CGP
        reloaded_model = tf.saved_model.load(STORAGE_LOCATION)
    return reloaded_model


if __name__ == '__main__':

    print("#### Loading the model ...")
    reloaded_model = get_model_from_gcp(local=True)
    print("#### Model uploaded ...")

    print("#### Predictions ...")

    text_samples = ["The FBI raided a “Cleveland office linked to Ukraine. Biden, Pelosi, \
        Kerry and Romney all had sons getting tens of millions of dollars \
        from no-show jobs in Ukraine.”",
        "Says the U.S. Senate is “dominated by millionaires” and that he \
        is “not one of them.”", "Says Kamala Harris called Joe Biden “a \
        racist” during a Democratic presidential debate.",
        "Donald Trump says he will ‘terminate’ Social Security if \
        re-elected.",
        "Say Joe Biden is a pedophile."]

    label_samples = ['fake', 'true', 'fake', 'fake', 'fake']

    for text_, label_ in zip(text_samples, label_samples):
        text_ = clean(text_)
        y_prob = reloaded_model([text_])
        print(f"Label: {label_} - Pred: {y_prob.numpy()[0][0]:.3f} - {text_[:100]}... ")
