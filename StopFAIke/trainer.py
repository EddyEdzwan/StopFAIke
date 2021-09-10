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
from StopFAIke.data import get_splits
from StopFAIke.params import BERT_MODEL_NAME
from StopFAIke.params import map_name_to_handle
from StopFAIke.params import map_model_to_preprocess


def init_TPU():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    return strategy


def get_model_name(BERT_MODEL_NAME):
    tfhub_handle_encoder = map_name_to_handle[BERT_MODEL_NAME]
    tfhub_handle_preprocess = map_model_to_preprocess[BERT_MODEL_NAME]

    print('BERT model selected           :', tfhub_handle_encoder)
    print('Preprocessing model auto-selected:', tfhub_handle_preprocess)
    return tfhub_handle_encoder, tfhub_handle_preprocess


def make_bert_preprocess_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    return tf.keras.Model(text_input, encoder_inputs)


def load_dataset(X, y, bert_preprocess_model, batch_size=32, is_training=True):
    X = [np.array([item]) for item in X]
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    num_examples = len(X)

    if is_training:
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda X_, y_: (bert_preprocess_model(X_), y_))
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return dataset, num_examples


def build_classifier_model():
    class Classifier(tf.keras.Model):
        def __init__(self):
            super(Classifier, self).__init__(name="prediction")
            self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True)
            self.dropout = tf.keras.layers.Dropout(0.5)
            self.dense_int = tf.keras.layers.Dense(128, activation='relu')
            self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid')

        def call(self, preprocessed_text):
            encoder_outputs = self.encoder(preprocessed_text)
            pooled_output = encoder_outputs["pooled_output"]
            x = self.dropout(pooled_output)
            x = self.dense_int(x)
            x = self.dropout(x)
            x = self.dense_final(x)
            return x

    model = Classifier()

    return model


def train_model(strategy, X_train, y_train, X_val, y_val):
    epochs = 1
    batch_size = 128
    init_lr = 3e-5

    print(f'Fine tuning {tfhub_handle_encoder} model')
    bert_preprocess_model = make_bert_preprocess_model()

    with strategy.scope():

        # Train dataset creation
        train_dataset, train_data_size = load_dataset(X_train, y_train,
                bert_preprocess_model, batch_size=batch_size, is_training=True)

        steps_per_epoch = train_data_size // batch_size
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = num_train_steps // 10

        # Validation dataset creation
        validation_dataset, validation_data_size = load_dataset(X_val, y_val,
                bert_preprocess_model, batch_size=batch_size, is_training=False)

        validation_steps = validation_data_size // batch_size

        # Model creation
        classifier_model = build_classifier_model()

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
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw')

        # Compile
        classifier_model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)

        # Training
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',
                patience=4, restore_best_weights=True)

        print("### Training ...")
        history = classifier_model.fit(
                    x=train_dataset,
                    validation_data=validation_dataset,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_steps=validation_steps,
                    callbacks=[es])

        print("### End of Training ...")

        return classifier_model


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


if __name__ == '__main__':
    # Init TPU
    strategy = init_TPU()

    # get training data from GCP bucket
    X, y = get_data()

    # train/val/test split
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(X, y)

    # Define model
    tfhub_handle_encoder, tfhub_handle_preprocess = get_model_name(BERT_MODEL_NAME)

    # train model on GCP (TPU required)
    model = train_model(strategy, X_train, y_train, X_val, y_val)


    # # save trained model to GCP bucket (whether the training occured locally or on GCP)
    # save_model(reg)
