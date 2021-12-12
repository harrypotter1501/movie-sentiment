# preprocess data

import pandas as pd
import pickle
from ast import literal_eval

from nltk.sentiment import SentimentIntensityAnalyzer

from utils.models import reviews_lda

from config import *


def sia(df):
    sia = SentimentIntensityAnalyzer()
    rev = pd.DataFrame(df['reviews'].apply(sia.polarity_scores).to_list())
    rev.reset_index(drop=True)
    return df.join(rev)


def read_and_process(path=path):
    df = pd.read_csv(path, index_col=False)

    df['genres'] = df['genres'] \
        .apply(lambda x: pickle.loads(literal_eval(x))) \
        .apply(lambda x: [g['name'].lower() for g in x])

    df['reviews_tmdb'] = df['reviews_tmdb'] \
        .apply(lambda x: pickle.loads(literal_eval(x))) \
        .apply(lambda x: ''.join([r['content'] for r in x]))

    df['reviews_imdb'] = df['reviews_imdb'] \
        .apply(lambda x: pickle.loads(literal_eval(x))) \
        .apply(lambda x: ''.join([r['content'] for r in x]))

    df['reviews'] = df[['reviews_tmdb', 'reviews_imdb']].agg(''.join, axis=1)

    return df.drop(['reviews_tmdb', 'reviews_imdb'], axis=1)


def preprocess(path=path):
    df = read_and_process(path)
    df = sia(df)
    df = reviews_lda(df)
    return df

