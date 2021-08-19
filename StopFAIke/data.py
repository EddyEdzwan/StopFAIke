import glob
import json
import pandas as pd
import numpy as np

import tldextract

from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize


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

def clean(text):
    """
    Provided by Le Wagon - Machine Learning - NLP
    Preprocessing articles - punctuation / lowercased / tokenize / stop_words / lemmatize
    """
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation
    lowercased = text.lower() # Lower Case
    tokenized = word_tokenize(lowercased) # Tokenize
    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
    stop_words = set(stopwords.words('english')) # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    lemma = WordNetLemmatizer() # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
    return lemmatized
