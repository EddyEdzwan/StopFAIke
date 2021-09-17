
"""Preprocessing functions"""

import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa

from StopFAIke.params import BERT_MODEL_NAME
from StopFAIke.params import map_name_to_handle
from StopFAIke.params import map_model_to_preprocess

AUTOTUNE = tf.data.experimental.AUTOTUNE


# def define_strategy():
#     """
#     Define strategy for model training:
#     - TPU strongly recommanded
#     - GPU very slow (1hr/epoch on full dataset)
#     - CPU not recommended
#     """
#     if tf.config.list_logical_devices('TPU'):
#         cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
#         tf.config.experimental_connect_to_cluster(cluster_resolver)
#         tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
#         strategy = tf.distribute.TPUStrategy(cluster_resolver)
#         print('Using TPU')
#     elif tf.config.list_logical_devices('GPU'):
#         strategy = tf.distribute.MirroredStrategy()
#         print('Using GPU')
#     else:
#         strategy = tf.distribute.get_strategy()
#         print('Using CPU (not recommended).')
#     return strategy


def define_strategy():
    """
    Detect automatically which processor units are available and define in function
    which tensorflow strategy is best for training:
        - TPU strongly recommanded
        - Multiple GPUs - Not available on Google Colab
        - One GPU - very slow (1hr/epoch on full dataset)
        - CPU - Not recommended

    Returns:
        Tensorflow strategy object
    """

    # Detect hardware
    try:
        tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    except ValueError:
        tpu_resolver = None
        gpus = tf.config.list_logical_devices("GPU")

    # Select appropriate distribution strategy
    if tpu_resolver:
        tf.config.experimental_connect_to_cluster(tpu_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
        print('Running on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])

        # Only for TPU, no link with startegy but needed to download BERT Preprocess / Encoder
        os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"

    elif len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('Running on single GPU ', gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('Running on CPU - Not recommended')

    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    return strategy


def get_model_name(BERT_MODEL_NAME=BERT_MODEL_NAME):
    """
    Define BERT model:
    - Preprocessing
    - Encoder

    Args:
    BERT_MODEL_NAME: string

    Returns:
        tfhub_handle_preprocess: BERT preprocess url for Tensoflow Hub
        tfhub_handle_encoder: BERT encoder url for Tensoflow Hub

    """

    tfhub_handle_preprocess = map_model_to_preprocess[BERT_MODEL_NAME]
    tfhub_handle_encoder = map_name_to_handle[BERT_MODEL_NAME]
    return tfhub_handle_preprocess, tfhub_handle_encoder


def make_bert_preprocess_model(tfhub_handle_preprocess):
    """
    Build BERT preprocessing Tensorflow model

    Args:
        tfhub_handle_preprocess: url for Tensoflow Hub

    Returns:
        BERT preprocessing Tensorflow model (not including in the model for
        training time reduction purpose)
    """

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    return tf.keras.Model(text_input, encoder_inputs)


def load_dataset(X, y, bert_preprocess_model, batch_size=32, is_training=True):
    """
    Create Tensorflow datasets

    Args:
        X: pandas Series, text data
        y: pandas Series, label data
        bert_preprocess_model: BERT preprocessing Tensorflow model instance
        batch_size: size of the batch
        is_training: True only for training dataset

    Returns:
        dataset: optimized Tensorflow dataset
        num_examples: number of samples (required for training parameters)
    """

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
