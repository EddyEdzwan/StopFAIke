
"""Helper functions"""

import string
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Required only for Colab
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean(text, stopword=False, lemmat=False):
    """
    Apply cleaning to data (text)
        - Remove Punctuation
        - Lower Case
        - Tokenization
        - Remove numbers
        - Remove stopwords (if needed)
        - Lemmatization (if needed)

    Args:
        text: string
        stopword: by default False (does not remove stop words)
        lemmat: by default False (does not apply lemmatization)

    Returns:
        clean_text: string
    """

    # Remove Punctuation
    for punctuation in string.punctuation:
        text.replace(punctuation, ' ')

    # Lower Case
    lowercased = text.lower()

    # Tokenize
    tokenized = word_tokenize(lowercased)

    # Remove numbers
    words = [word for word in tokenized if word.isalpha()]

    # Remove Stop Words
    if stopword:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if not word in stop_words]

    # Lemmatize
    if lemmat:
        lemma = WordNetLemmatizer()
        words = [lemma.lemmatize(word) for word in words]
    return ' '.join(word for word in words)


def drop_prefix(text, prefix='(Reuters)', n=5):
    """
    Returns string without prefix

    Args:
        text: string
        prefix: string
        n:

    Returns:
        clean_text: string

    """
    ts = str.split(text,' ')
    if prefix in ts[:n]:
        return str.split(text, prefix)[-1]
    else:
        return text


def plot_loss(history, title=None):
    """
    Plotting history model training

    Args:
        history: history from Tensorflow model fit method
        title: string
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Model loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylim(ymin=0, ymax=1)
    ax1.legend(['Train', 'Validation'], loc='best')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.set_title('ACC')
    ax2.set_ylabel('ACC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(ymin=0, ymax=1)
    ax2.legend(['Train', 'Validation'], loc='best')

    ax3.plot(history.history['recall'])
    ax3.plot(history.history['val_recall'])
    ax3.set_title('Recall')
    ax3.set_ylabel('Recall')
    ax3.set_xlabel('Epoch')
    ax3.set_ylim(ymin=0, ymax=1)
    ax3.legend(['Train', 'Validation'], loc='best')
    if title:
        fig.suptitle(title)
    plt.show()


def binary_metrics(y_test, y_pred):
    """
    Return binary metrics:
        - Accuracy score
        - Recall score
        - Precision score
        - f1 score


    Args:
        y_test: true labels
        y_pred: predicted labels
    """

    print('-'*80)
    print('Acc: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_pred)))
    print('Precision: {:.2f}'.format(precision_score(y_test, y_pred)))
    print('f1: {:.2f}'.format(f1_score(y_test, y_pred)))
    print('-'*80)


def get_metrics_ds(y_test, ds, model):
    """
    Returns metrics for model evaluation

    Args:
        y_test: true label
        ds: Tensorflow dataset (including X_test preprocessed)
        model: Tensorflow model

    """

    y_prob = model.predict(ds)
    y_pred = np.where(y_prob > 0.5, 1, 0)

    conf_matrix = confusion_matrix(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('-'*80)
    print(f"acc: {acc*100:.2f}%")
    print(f"recall: {recall*100:.2f}%")
    print(f"precision: {precision*100:.2f}%")
    print(f"f1: {f1*100:.2f}%")
    print('-'*80)

    sns.heatmap(conf_matrix, annot=True, fmt="d");


################
#  DECORATORS  #
################
def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int(te - ts)
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed
