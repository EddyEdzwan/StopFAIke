
"""Get data, cleaning, preprocessing"""

import pandas as pd

from sklearn.model_selection import train_test_split

from StopFAIke.utils import clean

from StopFAIke.params import BUCKET_NAME
from StopFAIke.params import P_TRAIN_DATA_PATH
from StopFAIke.params import FNN_TRAIN_DATA_PATH
from StopFAIke.params import BIS_T_TRAIN_DATA_PATH
from StopFAIke.params import BIS_F_TRAIN_DATA_PATH
from StopFAIke.params import PO_TRAIN_DATA_PATH

# path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/politifact_scrap.csv'
# path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/FakesNewsNET.csv'
# true_path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/True.csv'
# fake_path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/Fake.csv'
# path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/poynter_final_condensed.csv'
# df = pd.read_csv(path, nrows=nrows)


def get_data_from_gcp(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH):
    """
    Function to get the training data from GCP Storage

    Args:
        BUCKET_NAME: string, Google Storage Bucket name
        BUCKET_TRAIN_DATA_PATH: string, Google Storage Bucket subdirectory name

    Returns:
        DataFrame
    """

    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
    return df


def preprocess_bisaillon(true_df, fake_df):
    """
    Merge Bisaillon datasets (true + fake)

    Args:
        true_df DataFrame: news artciles labeled TRUE
        fake_df DataFrame: news artciles labeled TRUE

    Returns:
        Merged DataFrame
    """

    true_df.drop_duplicates(inplace=True)
    fake_df.drop_duplicates(inplace=True)

    true_df['category'] = 0
    fake_df['category'] = 1
    return pd.concat([true_df, fake_df]).reset_index(drop=True)


def get_data(nrows=100_000):
    """
    Merge all datasets:
    - Politifact dataset: from scrapping - X (fact) / y (label)
    - FakeNewsNET: from github repository - X (title) / y(label)
    - Bisaillon: from Kaggle - X(title) / y(label)
    - Poynter: from scrapping - X (fact) / y (label)

    Args:
        nrows: number of lines to keep

    Returns:
        X: pandas Series, text data
        y: pandas Series, label data
    """

    # Politifact
    P_df = get_data_from_gcp(BUCKET_NAME, P_TRAIN_DATA_PATH)
    X_P = P_df['statement'].copy()
    y_P = P_df['category2'].copy()

    # FakeNewsNET
    FNN_df = get_data_from_gcp(BUCKET_NAME, FNN_TRAIN_DATA_PATH)
    X_FNN = FNN_df['title'].copy()
    y_FNN = FNN_df['category'].copy()

    # Bisaillon
    BIS_T_df = get_data_from_gcp(BUCKET_NAME, BIS_T_TRAIN_DATA_PATH)
    BIS_F_df = get_data_from_gcp(BUCKET_NAME, BIS_F_TRAIN_DATA_PATH)
    BIS_df = preprocess_bisaillon(BIS_T_df, BIS_F_df)
    X_BIS = BIS_df['title'].copy()
    y_BIS = BIS_df['category'].copy()

    # Poynter
    PO_df = get_data_from_gcp(BUCKET_NAME, PO_TRAIN_DATA_PATH)
    X_PO = PO_df['title_list'].copy()
    y_PO = PO_df['label_list_transformed'].copy()

    # Merging All
    X = pd.concat([X_P, X_FNN, X_BIS, X_PO]).sample(frac=1, random_state=42).sample(n=nrows, random_state=42).reset_index(drop=True)
    y = pd.concat([y_P, y_FNN, y_BIS, y_PO]).sample(frac=1, random_state=42).sample(n=nrows, random_state=42).reset_index(drop=True)

    return X, y


def clean_data(X, y, stopword=False, lemmat=False):
    """
    Apply cleaning to dataset

    Args:
        X: pandas Series (text)
        y: pandas Series (label)
        stopword: False (does not remove stop words)
        lemmat: False (does not apply lemmatization)

    Returns:
        X_clean: pandas Series, cleaned text data
        y as pandas Series, label data
    """
    X_clean = X.apply(clean, stopword=stopword, lemmat=lemmat)
    return X_clean, y


def get_splits(X, y, valtest_size=0.3):
    """
    Function to split data in Train/Val/Test

    Args:
        X: pandas Series (text)
        y: pandas Series (label)
        valtest_size: size (in fraction) dedicated for val and test sets.

    Returns:
        X_train, y_train: pandas Series, training data
        X_val, y_val: pandas Series, val data
        X_test, y_test: pandas Series, test data
    """

    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=valtest_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':

    # get data from GCP bucket
    X, y = get_data(nrows=1_000)

    # clean data
    X_clean, y = clean_data(X, y, stopword=False, lemmat=False)

    # train/val/test split
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(X_clean, y, valtest_size=0.3)

    print('-'*80)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print('-'*80)
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print('-'*80)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print('-'*80)
    print(f"Fake (1) proportion in train (%): {y_train.sum()/len(y_train)*100:.3f}")
    print(f"Fake (1) proportion in val (%): {y_val.sum()/len(y_val)*100:.3f}")
    print(f"Fake (1) proportion in test (%): {y_test.sum()/len(y_test)*100:.3f}")
    print('-'*80)
