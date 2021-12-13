# models

import pandas as pd
import pyspark
from pyspark.sql import SQLContext

from nltk.corpus import stopwords
import re as re
from pyspark.ml.feature import CountVectorizer, IDF, VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression

from pyspark.ml.clustering import LDA
from pyspark.sql.types import (
    StructField, ArrayType, StructType, StringType, IntegerType, LongType, FloatType
)

from utils.context import SprkCxt

from config import *


def lda(text, num_topics, max_iterations, wordNumbers):
    #sc = pyspark.SparkContext.getOrCreate()
    sc = SprkCxt()
    data = sc.parallelize(text)

    stopWords = stopwords.words('english')
    tokens = data \
        .map(lambda document: document.strip().lower()) \
        .map(lambda document: re.split(' ', document)) \
        .map(lambda word: [x for x in word if x.isalpha()]) \
        .map(lambda word: [x for x in word if len(x) > 2] ) \
        .map(lambda word: [x for x in word if x not in stopWords]) \
        .zipWithIndex()

    schema = StructType([
        StructField('list_of_words', ArrayType(StringType()), True), 
        StructField('index', IntegerType(), True)
    ])

    sqlContext = SQLContext(sc)
    df_txts = sqlContext.createDataFrame(tokens, schema=schema)

    cv = CountVectorizer(inputCol='list_of_words', outputCol='raw_features', vocabSize=500, minDF=1.0)
    cvmodel = cv.fit(df_txts)
    result_cv = cvmodel.transform(df_txts)

    idf = IDF(inputCol='raw_features', outputCol='features')
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

    def flatten(llst):
        return [l for lst in llst for l in lst]

    words = pd.DataFrame(
        [
            [flatten(lda(
                    rev, 
                    num_topics, max_iterations, wordNumbers
                ))] for rev in df['reviews']
        ], columns=['lda']
    )
    words.reset_index(drop=True)
    df = df.join(words)
    return df.drop(['reviews'], axis=1)


def load_data(df=None, path=pp_path):
    sc = SprkCxt()
    sqlContext = SQLContext(sc)

    schema = StructType([
        StructField('id', IntegerType(), False), 
        StructField('title', StringType(), False), 
        StructField('release_date', LongType(), False), 
        StructField('vote_average', FloatType(), False), 
        StructField('vote_count', IntegerType(), False), 
        StructField('genres', ArrayType(StringType()), False), 
        StructField('budget', LongType(), False), 
        StructField('popularity', FloatType(), False), 
        StructField('revenue', LongType(), False), 
        StructField('neg', FloatType(), False), 
        StructField('neu', FloatType(), False), 
        StructField('pos', FloatType(), False), 
        StructField('compound', FloatType(), False), 
        StructField('lda', ArrayType(StringType()), False)
    ])

    if df is not None:
        return sqlContext.createDataFrame(df, schema=schema)
    else:
        print('Unfinished method!!!')

    return sqlContext.read.options(header='true', schema=schema).csv(path)


def vectorize(df):
    cv_genres = CountVectorizer(inputCol='genres', outputCol='genres_vec', minDF=1, minTF=1)
    cvg_model = cv_genres.fit(df)
    df = cvg_model.transform(df)

    cv_lda = CountVectorizer(inputCol='lda', outputCol='lda_vec', minDF=1, minTF=1)
    cvl_model = cv_lda.fit(df)
    df = cvl_model.transform(df)

    va_model = VectorAssembler(
        inputCols=[
            'release_date', 'vote_count', 'genres_vec', 'budget', 'popularity', 
            'revenue', 'neg', 'neu', 'pos', 'compound', 'lda_vec'
        ], 
        outputCol='features'
    )
    df = va_model.transform(df)

    return df


def lr(df):
    glr = GeneralizedLinearRegression(
        labelCol='vote_average', 
        family='gaussian', link='identity', 
        maxIter=10, regParam=0.3
    )
    glr_model = glr.fit(df)
    pred = glr_model.transform(df)

    # summary = glr_model.summary
    # print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
    # print("T Values: " + str(summary.tValues))
    # print("P Values: " + str(summary.pValues))
    # print("Dispersion: " + str(summary.dispersion))
    # print("Null Deviance: " + str(summary.nullDeviance))
    # print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    # print("Deviance: " + str(summary.deviance))
    # print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    # print("AIC: " + str(summary.aic))
    # print("Deviance Residuals: ")
    # summary.residuals().show()

    return pred

