# analysis

# modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from ast import literal_eval
from wordcloud import WordCloud

# configs
from config import *


def read_predicted(path=pred_path):
    """read in predicted data as dataframe"""

    df = pd.read_csv('./data/predicts.csv')
    df['error'] = df['prediction'] - df['vote_average']

    return df


def plt_err_dist(df):
    """plot distribution of prediction error"""

    bins = pd.IntervalIndex.from_tuples([(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), 
                                     (1.0, 1.2), (1.2, 1.4), (1.4, 10)])
    hist = df['error'].groupby(pd.cut(np.abs(df['error']), bins)).count()
    hist.plot(kind='bar')

    plt.ylabel('count')
    plt.show()


def plt_clust_err(df, thresh):
    """plot cluster percentage error"""

    bar = df.groupby(np.abs(df['error']) < thresh).mean().drop('error', axis=1)
    bar = bar / bar.sum(axis=0)

    x = np.arange(len(bar.columns))
    plt.bar(x - 0.2, bar.iloc[1], 0.4, label = 'Good')
    plt.bar(x + 0.2, bar.iloc[0], 0.4, label = 'Bad')

    plt.xticks(x, bar.columns, rotation='vertical')
    plt.ylim((0, 1))
    plt.ylabel('percentage error')
    plt.legend()

    plt.show()


def desc_clust_err(df, thresh):
    """describe cluster percentage error"""

    df_good = df[np.abs(df['prediction'] - df['vote_average']) < thresh]
    df_bad = df[np.abs(df['prediction'] - df['vote_average']) >= thresh]
    res = (df_bad.describe() - df_good.describe()) / df_good.describe()

    return res


def plt_cloud(df, thresh):
    """plt wordcloud"""

    with open('./data/topics.txt', 'r') as f:
        topics = literal_eval(f.read())

    # calculate mean distribution of good and bad words
    topic_dist = np.array([[p for p in literal_eval(dist)] for dist in df['topics']])
    df_good = df[np.abs(df['prediction'] - df['vote_average']) < thresh]
    df_bad = df[np.abs(df['prediction'] - df['vote_average']) >= thresh]
    good_topics = np.mean(topic_dist[df_good.index, :], axis=0)
    bad_topics = np.mean(topic_dist[df_bad.index, :], axis=0)

    good_words = []
    bad_words = []
    mask = good_topics > bad_topics

    # assemble good and bad words
    for i, p in enumerate(good_topics[mask]):
        if(mask[i]):
            good_words.extend(topics[i] * int(p * 100))
        else:
            bad_words.extend(topics[i] * int(p * 100))

    for i, p in enumerate(good_topics):
        good_words.extend(topics[i] * 10)
        bad_words.extend(topics[i] * 10)

    # cloud
    wordcloud = WordCloud(
        width = 1600,
        height = 900, 
        background_color='white'
    ).generate(' '.join(good_words))
    fig = plt.figure(
        figsize = (40, 30)
    )
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')

    plt.tight_layout(pad=0)
    plt.show()

    wordcloud = WordCloud(
        width = 1600,
        height = 900, 
        background_color='white'
    ).generate(' '.join(bad_words))
    fig = plt.figure(
        figsize = (40, 30)
    )
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')

    plt.tight_layout(pad=0)
    plt.show()

