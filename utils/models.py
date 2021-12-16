# models

# import modules
import pandas as pd
from pyspark.sql import SQLContext

# preprocessing
from nltk.corpus import stopwords
import re as re
from pyspark.ml.feature import CountVectorizer, IDF, VectorAssembler
from pyspark.ml.clustering import LDA
from pyspark.sql.types import (
    StructField, ArrayType, StructType, StringType, IntegerType, LongType, FloatType
)

from ast import literal_eval

# ml models
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# context manager
from utils.context import SprkCxt

# configs
from config import *


def lda(text, i, num_topics, max_iterations, wordNumbers):
    """
    Implement LDA on review text. 
    """

    sc = SprkCxt()
    data = sc.parallelize(text)

    # filter & transform -> list of words
    stopWords = stopwords.words('english')
    tokens = data \
        .map(lambda document: document.strip().lower()) \
        .map(lambda document: re.split(' ', document)) \
        .map(lambda word: [x for x in word if x.isalpha()]) \
        .map(lambda word: [x for x in word if len(x) > 2] ) \
        .map(lambda word: [x for x in word if x not in stopWords]) \
        .zipWithIndex()

    # dataframe schema & creation
    schema = StructType([
        StructField('list_of_words', ArrayType(StringType()), True), 
        StructField('index', IntegerType(), True)
    ])

    sqlContext = SQLContext(sc)
    df_txts = sqlContext.createDataFrame(tokens, schema=schema)

    # vectorizing
    cv = CountVectorizer(inputCol='list_of_words', outputCol='raw_features', vocabSize=50, minDF=2.0)
    cvmodel = cv.fit(df_txts)
    result_cv = cvmodel.transform(df_txts)

    # tf & idf counting
    idf = IDF(inputCol='raw_features', outputCol='features')
    idfModel = idf.fit(result_cv)
    result_tfidf = idfModel.transform(result_cv)

    # lda
    lda_model = LDA(k=num_topics)
    lda_model.setMaxIter(max_iterations)
    model = lda_model.fit(result_tfidf)

    # get topic words (visualization)
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
    """
    Implement LDA on review column. 
    """

    def flatten(llst):
        """flatten list of lists"""
        return [l for lst in llst for l in lst]

    words = pd.DataFrame(
        [
            [flatten(lda(
                    rev, i, 
                    num_topics, max_iterations, wordNumbers
                ))] for i, rev in enumerate(df['reviews'])
        ], columns=['lda']
    )
    words.reset_index(drop=True)
    df = df.join(words)
    return df.drop(['reviews'], axis=1)


def load_data(path=pp_path):
    """
    Load preprocessed data as spark dataframe.
    """

    sc = SprkCxt()
    sqlContext = SQLContext(sc)

    # define schema
    schema = StructType([
        StructField('id', IntegerType(), False), 
        StructField('title', StringType(), False), 
        StructField('release_date', LongType(), False), 
        StructField('vote_average', FloatType(), False), 
        StructField('vote_count', IntegerType(), False), 
        StructField('genres', ArrayType(StringType()), False), 
        #StructField('genres', StringType(), False), 
        StructField('budget', LongType(), False), 
        StructField('popularity', FloatType(), False), 
        StructField('revenue', LongType(), False), 
        StructField('neg', FloatType(), False), 
        StructField('neu', FloatType(), False), 
        StructField('pos', FloatType(), False), 
        StructField('compound', FloatType(), False), 
        StructField('lda', ArrayType(StringType()), False)
        #StructField('lda', StringType(), False)
    ])

    df = pd.read_csv(path)
    df = df.mask(df == 0).fillna(df.median()).astype(
        {'release_date': 'int64', 'vote_count': 'int32', 'budget': 'int64', 'revenue': 'int64'}
    )

    # rebuild list objects
    df['genres'] = df['genres'].apply(literal_eval)
    df['lda'] = df['lda'].apply(literal_eval)

    return sqlContext.createDataFrame(df, schema=schema)


def vectorize(df):
    """
    Vectorize string data and assemble the feature column.
    """
    
    # vectorize genres
    cv_genres = CountVectorizer(inputCol='genres', outputCol='genres_vec', minDF=1, minTF=1)
    cvg_model = cv_genres.fit(df)
    df = cvg_model.transform(df)

    # vectorize lda
    cv_lda = CountVectorizer(inputCol='lda', outputCol='lda_vec', minDF=2, minTF=2)
    cvl_model = cv_lda.fit(df)
    df = cvl_model.transform(df)

    # assemble features
    va_model = VectorAssembler(
        inputCols=[
            'vote_count', 'genres_vec', 'budget', 
            'neg', 'neu', 'pos', 'compound', #'lda_vec'
        ], 
        outputCol='features'
    )
    df = va_model.transform(df)

    return df


def validate(data_path=pp_path):
    """
    Train a model on validate it. 
    """

    df = load_data(data_path)
    df = vectorize(df)

    # split for train and validation
    print('Total samples: {}'.format(df.count()))
    train_df, test_df = df.randomSplit([0.8, 0.2])

    # split by score
    N = train_df.count()

    # define model
    glr = GeneralizedLinearRegression(
        labelCol='vote_average', 
        family='gaussian', link='identity', 
        weightCol='vote_count', 
        maxIter=50, regParam=0.05
    )

    # fit on training set
    print('Train data size: {}'.format(N))
    glr_model = glr.fit(train_df)

    # train summary
    summary = glr_model.summary
    #print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
    #print("T Values: " + str(summary.tValues))
    #print("P Values: " + str(summary.pValues))
    print("\tDispersion: " + str(summary.dispersion))
    print("\tNull Deviance: " + str(summary.nullDeviance))
    print("\tResidual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    print("\tDeviance: " + str(summary.deviance))
    print("\tResidual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    print("\tAIC: " + str(summary.aic))
    #print("Deviance Residuals: ")
    #summary.residuals().show()'''

    # evaluate on test set
    print('Test data size: {}'.format(test_df.count()))
    pred_glr = glr_model.transform(test_df)
    evl = RegressionEvaluator(labelCol='vote_average')
    rmse = evl.evaluate(pred_glr)
    print('GLR Evaluation RMSE: {}'.format(rmse))
    #return glr_model

    # extract feature imporatance
    #print(ExtractFeatureImp(glr_model.stages[-1].featureImportances, pred_glr, "features").head(10))

    # save test results
    pred_glr.select(
        ['title', 'prediction', 'vote_average', 'popularity', 
         'release_date', 'vote_count', 'budget', 'revenue', 'neg', 'lda']
    ).toPandas().to_csv('./data/predicts.csv', index=False)
    print('Results saved to ./data/predicts.csv')

    return pred_glr


