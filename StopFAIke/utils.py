import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize

# Required only for Colab
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# TODO : include parameters to select Lemmatize, stop words ...
def clean(text):

    # Remove Punctuation
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')

    # Lower Case
    lowercased = text.lower()

    # Tokenize
    tokenized = word_tokenize(lowercased)

    # Remove numbers
    words_only = [word for word in tokenized if word.isalpha()]

    # Make stopword list
    # stop_words = set(stopwords.words('english'))

    # Remove Stop Words
    # without_stopwords = [word for word in words_only if not word in stop_words]

    # Lemmatize
    # lemma = WordNetLemmatizer() # Initiate Lemmatizer
    # lemmatized = [lemma.lemmatize(word) for word in without_stopwords]
    return ' '.join(word for word in words_only)
