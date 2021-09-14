import pandas as pd

import tensorflow as tf
import tensorflow_text as text

from pathlib import Path
from google.cloud import storage

from StopFAIke.params import BUCKET_NAME, STORAGE_LOCATION

# root_dir = os.path.dirname(os.path.dirname(__file__))
# LOCAL_PATH = os.path.join(root_dir, "raw_data", "test.csv")


def get_model_from_gcp(local=False):
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

    # print("#### Predictions ...")
    # samples = ['Julien is 43yo',
    #            'Nina and Fleur are the daughters of Julien',
    #            'Trump president']

    # for sample in samples:
    #     y_prob = reloaded_model([sample])
    #     print(f"Pred: {y_prob.numpy()[0][0]:.3f} - {sample} ")

    # print("#### Predictions ...")
    # sample = ['Julien is 43yo']

    # y_prob = reloaded_model(sample)
    # print(f"Pred: {y_prob.numpy()[0][0]:.3f} - {sample} ")

    print("#### Predictions ...")
    X = pd.DataFrame(dict(text=['Julien is 43yo']))

    # Get the model (locally or from GCP)
    reloaded_model = get_model_from_gcp(local=True)

    # Make prediction
    y_prob = reloaded_model(X['text'].values.tolist())
    print(f"Pred: {y_prob.numpy()[0][0]:.3f} - {X['text'].values.tolist()} ")
