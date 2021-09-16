
import os

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa

from official.nlp import optimization

from google.cloud import storage

from StopFAIke.data import get_data
from StopFAIke.data import clean_data
from StopFAIke.data import get_splits

from StopFAIke.params import BERT_MODEL_NAME
# from StopFAIke.params import map_name_to_handle
# from StopFAIke.params import map_model_to_preprocess

from StopFAIke.preprocessor import get_model_name
from StopFAIke.preprocessor import define_strategy
from StopFAIke.preprocessor import make_bert_preprocess_model
from StopFAIke.preprocessor import load_dataset

from StopFAIke.network import build_classifier_model

from StopFAIke.utils import simple_time_tracker
from StopFAIke.utils import plot_loss
from StopFAIke.utils import get_metrics_ds

AUTOTUNE = tf.data.experimental.AUTOTUNE    # (AUTOTUNE = tf.data.AUTOTUNE)


class Trainer:
    def __init__(self, BERT_MODEL_NAME=BERT_MODEL_NAME, epochs=1, batch_size=32):
        self.X = None
        self.y = None
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
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_lr = 3e-5
        self.classifier_model = None
        self.history = None

    @simple_time_tracker
    def load_data(self, nrows=1_000):
        print("###### Loading data....")
        self.X = get_data(nrows=nrows)[0]
        self.y = get_data(nrows=nrows)[1]

    @simple_time_tracker
    def clean(self):
        print("###### cleaning....")
        self.X = clean_data(self.X, self.y)[0]
        self.y = clean_data(self.X, self.y)[1]

    @simple_time_tracker
    def preproc(self, valtest_size=0.3):
        print("###### preprocessing....")
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = get_splits(self.X, self.y, valtest_size=valtest_size)

        # self.pipe = create_pipeline()
        # self.X_train_preproc = self.pipe.fit_transform(self.X_train)
        print('-'*80)
        print(f"###### shape of X_train, y_train: {self.X_train.shape, self.y_train.shape}")
        print(f"###### shape of X_val, y_val: {self.X_val.shape, self.y_val.shape}")
        print(f"###### shape of X_test, y_test: {self.X_test.shape, self.y_test.shape}")
        print('-'*80)

    @simple_time_tracker
    def train_model(self, plot_history=True):
        """
        Traing BERT model on Google Colab (TPU strongly recommended)
        """

        print(f'###### Model training {self.tfhub_handle_encoder} model')
        self.strategy = define_strategy()

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
                        validation_data=self.validation_dataset,
                        steps_per_epoch=steps_per_epoch,
                        epochs=self.epochs,
                        validation_steps=validation_steps,
                        callbacks=[es])
            print("### End of Training ...")

            # Print & plot some key training results
            print("####### Train Accuracy", min(self.history.history["accuracy"]))
            print("####### Val Accuracy", min(self.history.history["val_accuracy"]))
            print("####### epochs reached", len(self.history.epoch))
            if plot_history:
                plot_loss(self.history, title='bert model - fine-tuning')

    @simple_time_tracker
    def evaluate_model(self, X_test=None, y_test=None):
        """
        Evaluates the model on a test set.
        Return the metrics: Acc, Recall, Precision, f1 score
        """
        # If no test set is given, use the holdout from train/test/split
        print("###### evaluating the model on a test set...")
        X_test = X_test or self.X_test
        y_test = y_test or self.y_test

        self.test_dataset, _ = load_dataset(X_test, y_test, self.bert_preprocess_model, batch_size=self.batch_size, is_training=False)

        get_metrics_ds(y_test, self.test_dataset, self.classifier_model)

    @simple_time_tracker
    def save_model(self):
        """
        Saving model to Google Colab
        """
        main_save_path = './my_models'
        saved_model_name = 'my_bert_model'

        saved_model_path = os.path.join(main_save_path, saved_model_name)

        preprocess_inputs = self.bert_preprocess_model.inputs
        bert_encoder_inputs = self.bert_preprocess_model(preprocess_inputs)
        bert_outputs = self.classifier_model(bert_encoder_inputs)
        model_for_export = tf.keras.Model(preprocess_inputs, bert_outputs)

        print('###### saving model....', saved_model_path)
        # Save everything on the Colab host (even the variables from TPU memory)
        save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
        model_for_export.save(saved_model_path, include_optimizer=False, options=save_options)


if __name__ == '__main__':

    # Instanciate trainer with number of rows to download and use
    trainer = Trainer(BERT_MODEL_NAME=BERT_MODEL_NAME, epochs=1, batch_size=32)

    # Load data with number of rows to download and use
    trainer.load_data(nrows=1_000)

    # Clean data
    trainer.clean()

    # Create train/test/split
    trainer.preproc(valtest_size=0.3)

    # Train BERT model and show training performance
    trainer.train_model(plot_history=True)

    # Evaluate on test set (by default the holdout from train/test/split)
    trainer.evaluate_model(X_test=None, y_test=None)

    # Save model on Google Colab
    # trainer.save_model()
