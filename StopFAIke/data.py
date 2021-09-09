# import numpy as np
import pandas as pd

# from google.cloud import storage

from StopFAIke.utils import clean
from StopFAIke.params import BUCKET_NAME
from StopFAIke.params import P_TRAIN_DATA_PATH
from StopFAIke.params import FNN_TRAIN_DATA_PATH
from StopFAIke.params import BIS_T_TRAIN_DATA_PATH
from StopFAIke.params import BIS_F_TRAIN_DATA_PATH
from StopFAIke.params import PO_TRAIN_DATA_PATH

from sklearn.model_selection import train_test_split

# path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/politifact_scrap.csv'
# path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/FakesNewsNET.csv'
# true_path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/True.csv'
# fake_path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/Fake.csv'
# path = 'https://storage.googleapis.com/wagon-data-615-seguy/data/poynter_final_condensed.csv'
# df = pd.read_csv(path, nrows=nrows)


def get_data_from_gcp(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, nrows=20000):
    """method to get the training data (or a portion of it) from GCP"""
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=nrows)
    return df


def preprocess_bisaillon(true_df, fake_df):
    true_df.drop_duplicates(inplace=True)
    fake_df.drop_duplicates(inplace=True)

    true_df['category'] = 0
    fake_df['category'] = 1

    return pd.concat([true_df, fake_df]).reset_index(drop=True)


def get_data(nrows=20000):
    # Politifact
    P_df = get_data_from_gcp(BUCKET_NAME, P_TRAIN_DATA_PATH, nrows=nrows)
    X_P = P_df['statement'].apply(clean).copy()
    y_P = P_df['category2'].copy()

    # FakeNewsNET
    FNN_df = get_data_from_gcp(BUCKET_NAME, FNN_TRAIN_DATA_PATH, nrows=nrows)
    X_FNN = FNN_df['title'].apply(clean).copy()
    y_FNN = FNN_df['category'].copy()

    # Bisaillon
    BIS_T_df = get_data_from_gcp(BUCKET_NAME, BIS_T_TRAIN_DATA_PATH, nrows=nrows)
    BIS_F_df = get_data_from_gcp(BUCKET_NAME, BIS_F_TRAIN_DATA_PATH, nrows=nrows)
    BIS_df = preprocess_bisaillon(BIS_T_df, BIS_F_df)
    X_BIS = BIS_df['title'].apply(clean).copy()
    y_BIS = BIS_df['category'].copy()

    # Poynter
    PO_df = get_data_from_gcp(BUCKET_NAME, PO_TRAIN_DATA_PATH, nrows=nrows)
    X_PO = PO_df['title_list'].apply(clean).copy()
    y_PO = PO_df['label_list_transformed'].copy()

    # Merging All
    X = pd.concat([X_P, X_FNN, X_BIS, X_PO]).sample(frac=1, random_state=42).reset_index(drop=True)
    y = pd.concat([y_P, y_FNN, y_BIS, y_PO]).sample(frac=1, random_state=42).reset_index(drop=True)

    return X, y


def get_splits(X, y):
    """method to split data in Train/Val/Test"""
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    # get training data from GCP bucket
    X, y = get_data()

    # train/val/test split
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(X, y)

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
