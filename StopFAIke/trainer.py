import numpy as np
import pandas as pd

import os
import tensorflow as tf
# AUTOTUNE = tf.data.AUTOTUNE
AUTOTUNE = tf.data.experimental.AUTOTUNE

import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa
from official.nlp import optimization

import matplotlib.pyplot as plt

from google.cloud import storage

from StopFAIke.data import get_data
from StopFAIke.data import clean_data
from StopFAIke.data import get_splits
from StopFAIke.params import BERT_MODEL_NAME
from StopFAIke.params import map_name_to_handle
from StopFAIke.params import map_model_to_preprocess

from StopFAIke.preprocessor import get_model_name
from StopFAIke.preprocessor import get_strategy
from StopFAIke.preprocessor import make_bert_preprocess_model
from StopFAIke.preprocessor import load_dataset

from StopFAIke.network import build_classifier_model


class Trainer:
    def __init__(self, nrows=1_000, BERT_MODEL_NAME=BERT_MODEL_NAME):
        self.X = get_data(nrows=nrows)[0]
        self.y = get_data(nrows=nrows)[1]
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.tfhub_handle_preprocess = get_model_name(BERT_MODEL_NAME)[0]
        self.tfhub_handle_encoder = get_model_name(BERT_MODEL_NAME)[1]
        self.bert_preprocess_model = None
        self.strategy = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.epochs = 1
        self.batch_size = 128
        self.init_lr = 3e-5
        # self.X_train_preproc = None
        # self.pipe = None
        self.classifier_model = None
        self.history = None

    def clean(self):
        print("###### cleaning....")
        self.X = clean_data(self.X, self.y)[0]
        self.y = clean_data(self.X, self.y)[1]

    def preproc(self, valtest_size=0.3):
        print("###### preprocessing....")
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = get_splits(self.X, self.y, valtest_size=valtest_size)

        # self.pipe = create_pipeline()
        # self.X_train_preproc = self.pipe.fit_transform(self.X_train)
        print("###### shape of X_train, y_train:")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print("###### shape of X_val, y_train:")
        print(f"X_train shape: {self.X_val.shape}")
        print(f"X_train shape: {self.X_val.shape}")
        print("###### shape of X_train, y_train:")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"X_test shape: {self.X_test.shape}")

    # @simple_time_tracker
    # def fit(self, plot_history=True, verbose=1):
    #     print("###### fitting...")
    #     # TensorFlow cannot work with Sparse Matrix out of Sklearn's OHE
    #     self.X_train_preproc = self.X_train_preproc.todense()
    #     self.network = Network(input_dim=self.X_train_preproc.shape[1])
    #     print(self.network.model.summary())
    #     self.network.compile_model()
    #     self.history = self.network.fit_model(
    #         self.X_train_preproc, self.y_train, verbose=verbose
    #     )

    #     # Print & plot some key training results
    #     print("####### min val MAE", min(self.history.history["val_mae"]))
    #     print("####### epochs reached", len(self.history.epoch))
    #     if plot_history:
    #         plot_model_history(self.history)

    def train_model(self):
        print(f'###### Fine tuning {self.tfhub_handle_encoder} model')
        self.strategy = get_strategy()

        self.bert_preprocess_model = make_bert_preprocess_model(self.tfhub_handle_preprocess)

        with self.strategy.scope():

            # Train dataset creation
            self.train_dataset, train_data_size = load_dataset(self.X_train, self.y_train,
                    self.bert_preprocess_model, batch_size=self.batch_size, is_training=True)

            steps_per_epoch = train_data_size // self.batch_size
            num_train_steps = steps_per_epoch * self.epochs
            num_warmup_steps = num_train_steps // 10

            # Validation dataset creation
            self.validation_dataset, validation_data_size = load_dataset(self.X_val, self.y_val,
                    self.bert_preprocess_model, batch_size=self.batch_size, is_training=False)

            validation_steps = validation_data_size // self.batch_size

            # Model creation
            self.classifier_model = build_classifier_model(self.tfhub_handle_encoder)

            # Metrics
            METRICS = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ]

            # Loss function
            loss = tf.keras.losses.BinaryCrossentropy()

            # Optimizer
            optimizer = optimization.create_optimizer(
                init_lr=self.init_lr,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                optimizer_type='adamw')

            # Compile
            self.classifier_model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)

            # Training
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',
                    patience=4, restore_best_weights=True)

            print("### Training ...")
            self.history = self.classifier_model.fit(
                        x=self.train_dataset,
                        validation_data=validation_dataset,
                        steps_per_epoch=steps_per_epoch,
                        epochs=self.epochs,
                        validation_steps=validation_steps,
                        callbacks=[es])

            print("### End of Training ...")


# def upload_model_to_gcp():
#     client = storage.Client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(STORAGE_LOCATION)
#     blob.upload_from_filename('model.joblib')


# def save_model(reg):
#     """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
#     HINTS : use joblib library and google-cloud-storage"""

#     # saving the trained model to disk is mandatory to then beeing able to upload it to storage
#     # Implement here
#     joblib.dump(reg, 'model.joblib')
#     print("saved model.joblib locally")

#     # Implement here
#     upload_model_to_gcp()
#     print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

# def wait_for_tpu_cluster_resolver_ready():
#     """Waits for `TPUClusterResolver` to be ready and return it.

#     Returns:
#         A TPUClusterResolver if there is TPU machine (in TPU_CONFIG).
#         Otherwise, return None.
#     Raises:
#         RuntimeError: if failed to schedule TPU.
#     """
#     tpu_config_env = os.environ.get('TPU_CONFIG')
#     if not tpu_config_env:
#         tf.logging.info('Missing TPU_CONFIG, use CPU/GPU for training.')
#     return None

#     tpu_node = json.loads(tpu_config_env)
#     tf.logging.info('Waiting for TPU to be ready: \n%s.', tpu_node)

#     num_retries = 100
#     for i in range(num_retries):
#         try:
#             tpu_cluster_resolver = (
#                 tf.contrib.cluster_resolver.TPUClusterResolver(
#                       tpu=[tpu_node['tpu_node_name']],
#                       zone=tpu_node['zone'],
#                       project=tpu_node['project'],
#                       job_name='worker'))
#             tpu_cluster_resolver_dict = tpu_cluster_resolver.cluster_spec().as_dict()
#             if 'worker' in tpu_cluster_resolver_dict:
#                 tf.logging.info('Found TPU worker: %s', tpu_cluster_resolver_dict)
#             return tpu_cluster_resolver
#         except Exception as e:
#             if i < num_retries - 1:
#                 tf.logging.info('Still waiting for provisioning of TPU VM instance.')
#             else:
#                 # Preserves the traceback.
#                 raise RuntimeError('Failed to schedule TPU: {}'.format(e))
#         time.sleep(10)

#     # Raise error when failed to get TPUClusterResolver after retry.
#     raise RuntimeError('Failed to schedule TPU.')

if __name__ == '__main__':

    wait_for_tpu_cluster_resolver_ready()

    # Init TPU
    print('###### Init TPU ######')
    # resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    # tf.config.experimental_connect_to_cluster(resolver)
    # tf.tpu.experimental.initialize_tpu_system(resolver)
    # strategy = tf.distribute.experimental.TPUStrategy(resolver)
    print(f"strategy: {strategy}")
    print('###### Using TPU ######')

    # Init TPU
    # strategy = init_TPU()

    # get training data from GCP bucket
    # X, y = get_data()

    # # train/val/test split
    # X_train, y_train, X_val, y_val, X_test, y_test = get_splits(X, y)

    # # Define model
    # tfhub_handle_encoder, tfhub_handle_preprocess = get_model_name(BERT_MODEL_NAME)

    # # train model on GCP (TPU required)
    # model = train_model(strategy, X_train, y_train, X_val, y_val)


    # # save trained model to GCP bucket (whether the training occured locally or on GCP)
    # save_model(reg)
