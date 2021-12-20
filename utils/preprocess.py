# preprocess data

# moduels
import pandas as pd
import pickle
from ast import literal_eval
from functools import reduce
import datetime

from nltk.sentiment import SentimentIntensityAnalyzer

# preprocess method
from utils.models import reviews_lda

# configs
from config import *


def read_and_process(path=path):
    """
    Read in raw data and do filtering.
    """

    df = pd.read_csv(path, index_col=False)

    def str_2_time(s):
        """convert datetime string to integer"""
        try:
            par = [int(t) for t in s.split('-')]
            return int(datetime.datetime(*par).timestamp())
        except Exception:
            return 0

    # process time -> int
    df['release_date'] = df['release_date'] \
        .apply(str_2_time)

    # process genres -> list of words
    df['genres'] = df['genres'] \
        .apply(lambda x: pickle.loads(literal_eval(x))) \
        .apply(lambda x: [g['name'].lower() for g in x])

    # process tmdb reviews -> texts
    df['reviews_tmdb'] = df['reviews_tmdb'] \
        .apply(lambda x: pickle.loads(literal_eval(x))) \
        .apply(lambda x: ' '.join([r['content'] for r in x]))

    # process imdb reviews -> texts
    df['reviews_imdb'] = df['reviews_imdb'] \
        .apply(lambda x: pickle.loads(literal_eval(x))) \
        .apply(lambda x: ' '.join([r['content'] for r in x]))

    def join(series):
        """join contents of multiple pandas series"""
        return reduce(lambda x,y: x + y, series)

    # join the reviews
    df['reviews'] = df[['reviews_tmdb', 'reviews_imdb']].agg(join, axis=1)

    return df.drop(['reviews_tmdb', 'reviews_imdb'], axis=1)


def sia(df):
    """
    Analyze sentiment intesity for all texts.
    """

    sia = SentimentIntensityAnalyzer()

    def sia_list(list):
        """join a list of strings"""
        return sia.polarity_scores(''.join(list))

    # create and join new dataframe with sentiment features
    rev = pd.DataFrame(df['reviews'].apply(sia_list).to_list())
    rev.reset_index(drop=True)

    return df.join(rev)


def preprocess(num_topics=num_topics, max_iterations=max_iterations, 
               wordNumbers=wordNumbers, path=path, pp_path=pp_path):
    """
    Preprocess raw data.
    Perform feature extraction, SIA & LDA.
    """

    # extraction
    print('Reading from {}...'.format(path))
    df = read_and_process(path)

    # sia
    print('Analyzing Sentiment...')
    df = sia(df)

    # lda
    print('Performing LDA...')
    df, topics = reviews_lda(df, num_topics=num_topics, 
                             max_iterations=max_iterations, wordNumbers=wordNumbers)

    # save topic words
    topics_path = './data/topics.txt'
    with open(topics_path, 'w') as f:
        f.write(str(topics))
    print('Topics persisted to {}'.format(topics_path))

    df.to_csv(pp_path, index=False)
    print('Data persisted to {}'.format(pp_path))

    return df

