import os
import glob
import json

import numpy as np
import pandas as pd

import tldextract


def get_data(directory):
    """
    Function to read news content in given directory using the FakeNewsNet dataset
    """
    dictlist = []
    cols = ['title', 'text', 'authors', 'num_images', 'domain', 'url']

    folders = glob.glob(directory + '/*')
    for index, subdir in enumerate(folders):

        file_path = glob.glob(subdir + '/*')

        #check if glob returned a valid file path (non-empty list)
        if len(file_path) == 1:
            file = open(file_path[0]).read()
            jsondata = json.loads(file)
            dictlist.append(scaledict(jsondata))
    return pd.DataFrame(dictlist, columns=cols)

def scaledict(ajson):
    """
    process json pulled from FakeNewsNet
    """
    thedict = {'url': ajson['url'],
                'title': ajson['title'],
                'text': ajson['text'],
                'num_images': len(ajson['images']),
                'authors': str(ajson['authors'])
                }

    ext = tldextract.extract(ajson['url'])
    thedict['domain'] = ext.domain
    return thedict

def remove_missing_values(df, col, exclude):
    """
    loops through df columns and drops values located in exclude variable, both can be single values
    """
    if type(col) == 'list':
        try:
            for ind, c in enumerate(col):
                indices = df[df[c] == exclude[ind]].index
                df = df.drop(indices)
        except:
            print('Exception occurred, check kwargs')
    else:
        indices = df[df[col] == exclude].index
        df = df.drop(indices)
    return df

def main(path_to_load, path_to_save):

    # Loading the data
    dir_pol_real = os.path.join(path_to_load, 'politifact', 'real')
    dir_pol_fake = os.path.join(path_to_load, 'politifact', 'fake')
    dir_gos_real = os.path.join(path_to_load, 'gossipcop', 'real')
    dir_gos_fake = os.path.join(path_to_load, 'gossipcop', 'fake')

    pol_real_df = get_data(dir_pol_real)
    pol_fake_df = get_data(dir_pol_fake)
    gos_real_df = get_data(dir_gos_real)
    gos_fake_df = get_data(dir_gos_fake)

    # Labeling
    pol_real_df['category'] = 8     #True
    pol_fake_df['category'] = 1     #Fake
    gos_real_df['category'] = 0
    gos_fake_df['category'] = 1

    pol_real_df['news_type'] = 'political'
    pol_fake_df['news_type'] = 'political'
    gos_real_df['news_type'] = 'gossip'
    gos_fake_df['news_type'] = 'gossip'

    # Merging
    data = pd.concat([pol_real_df, pol_fake_df, gos_real_df, gos_fake_df]).reset_index(drop=True)

    print('-'*80)
    print(f"data shape: {data.shape}")
    print('-'*80)
    print(f"ratio #true: {len(data[data['category']==0])/len(data)*100:.2f}%")
    print(f"ratio #fake: {len(data[data['category']==1])/len(data)*100:.2f}%")
    print(f"ratio #true - political: {len(data[(data['category']==0) & (data['news_type']=='political')])/len(data)*100:.2f}%")
    print(f"ratio #fake - political: {len(data[(data['category']==1) & (data['news_type']=='political')])/len(data)*100:.2f}%")
    print(f"ratio #true - gossip: {len(data[(data['category']==0) & (data['news_type']=='gossip')])/len(data)*100:.2f}%")
    print(f"ratio #fake - gossip: {len(data[(data['category']==1) & (data['news_type']=='gossip')])/len(data)*100:.2f}%")
    print('-'*80)

    # Missing values
    data = remove_missing_values(data, 'title', '')
    data = remove_missing_values(data, 'text', '')
    print(f"data shape (wo missing values): {data.shape}")

    # Duplicates
    data.drop_duplicates(inplace=True)
    print(f"data shape (wo duplicates): {data.shape}")

    # Saving to CSV
    data.to_csv(path_to_save, index=False)
    print("### CSV file created")


if __name__ == '__main__':

    path_to_load = '/Users/julienseguy/code/EddyEdzwan/StopFAIke/raw_data/fakenewsnet_dataset'
    path_to_save = '/Users/julienseguy/code/EddyEdzwan/StopFAIke/raw_data/FakesNewsNET.csv'

    main(path_to_load, path_to_save)
