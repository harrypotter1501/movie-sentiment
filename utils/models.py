# models

import pandas as pd
import pyspark
from pyspark.sql import SQLContext

from nltk.corpus import stopwords
import re as re
from pyspark.ml.feature import CountVectorizer , IDF

#from pyspark.mllib.linalg import Vector, Vectors
from pyspark.ml.clustering import LDA

from config import *


def lda(text, num_topics, max_iterations, wordNumbers):
    sc = pyspark.SparkContext.getOrCreate()
    data = sc.parallelize(text)

    stopWords = stopwords.words("english")
    tokens = data \
        .map(lambda document: document.strip().lower()) \
        .map(lambda document: re.split(" ", document)) \
        .map(lambda word: [x for x in word if x.isalpha()]) \
        .map(lambda word: [x for x in word if len(x) > 2] ) \
        .map(lambda word: [x for x in word if x not in stopWords]) \
        .zipWithIndex()

    sqlContext = SQLContext(sc)
    df_txts = sqlContext.createDataFrame(tokens, ["list_of_words",'index'])

    cv = CountVectorizer(inputCol="list_of_words", outputCol="raw_features", vocabSize=5000, minDF=1.0)
    cvmodel = cv.fit(df_txts)
    result_cv = cvmodel.transform(df_txts)

    idf = IDF(inputCol="raw_features", outputCol="features")
    idfModel = idf.fit(result_cv)
    result_tfidf = idfModel.transform(result_cv)

    lda_model = LDA(k=num_topics)
    lda_model.setMaxIter(max_iterations)
    model = lda_model.fit(result_tfidf)

    topicIndices = model.describeTopics(maxTermsPerTopic = wordNumbers)
    topics_final = topicIndices.select('termIndices').rdd.map(
        lambda topic: [i for i in topic.termIndices]
    ).collect()

    res = [
        [cvmodel.vocabulary[id] for id in topics_final[i]]
        for i in range(len(topics_final))
    ]

    return res


def reviews_lda(df, num_topics=3, max_iterations=10, wordNumbers=5):
    words = pd.DataFrame(
        [[l] for l in lda(
            df['reviews'], 
            num_topics, max_iterations, wordNumbers
        )], columns=['lda']
    )
    words.reset_index(drop=True)
    return df.join(words)

