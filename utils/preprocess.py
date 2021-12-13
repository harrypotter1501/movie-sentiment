# preprocess data

import pandas as pd
import pickle
from ast import literal_eval
from functools import reduce
import datetime

from nltk.sentiment import SentimentIntensityAnalyzer
from pyspark.sql import dataframe

from utils.models import reviews_lda

from config import *


def read_and_process(path=path):
    df = pd.read_csv(path, index_col=False)

    def str_2_time(s):
        par = [int(t) for t in s.split('-')]
        return int(datetime.datetime(*par).timestamp())

    df['release_date'] = df['release_date'] \
        .apply(str_2_time)

    df['genres'] = df['genres'] \
        .apply(lambda x: pickle.loads(literal_eval(x))) \
        .apply(lambda x: [g['name'].lower() for g in x])

    df['reviews_tmdb'] = df['reviews_tmdb'] \
        .apply(lambda x: pickle.loads(literal_eval(x))) \
        .apply(lambda x: [r['content'] for r in x])

    df['reviews_imdb'] = df['reviews_imdb'] \
        .apply(lambda x: pickle.loads(literal_eval(x))) \
        .apply(lambda x: [r['content'] for r in x])

    def join(series):
        return reduce(lambda x,y: x + y, series)

    df['reviews'] = df[['reviews_tmdb', 'reviews_imdb']].agg(join, axis=1)

    return df.drop(['reviews_tmdb', 'reviews_imdb'], axis=1)


def sia(df):
    sia = SentimentIntensityAnalyzer()

    def sia_list(list):
        return sia.polarity_scores(''.join(list))

    rev = pd.DataFrame(df['reviews'].apply(sia_list).to_list())
    rev.reset_index(drop=True)

    return df.join(rev)


def preprocess(path=path, pp_path=pp_path):
    print('Reading from {}...'.format(path))
    df = read_and_process(path)

    print('Analyzing Sentiment...')
    df = sia(df)

    print('Performing LDA...')
    df = reviews_lda(df)

    df.to_csv(pp_path, index=False)
    print('Data persisted to {}'.format(pp_path))

    return df

