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
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# context manager
from utils.context import SprkCxt

# configs
from config import *


def reviews_lda(df, num_topics=3, max_iterations=10, wordNumbers=5):
    """
    Implement LDA on review column. 
    """

    sc = SprkCxt()
    
    stopWords = stopwords.words('english')

    df['reviews'] = df['reviews'] \
            .apply(lambda line: line.split()) \
            .apply(lambda l: [x for x in l if x.isalpha() and len(x) > 2]) \
            .apply(lambda l: [x for x in l if x not in stopWords])

    schema = StructType([
        StructField('list_of_words', ArrayType(StringType()), True), 
        #StructField('index', IntegerType(), True)
    ])
    
    sqlContext = SQLContext(sc)
    sdf = sqlContext.createDataFrame(df[['reviews']], schema=schema)

    # dataframe schema & creation
    cv = CountVectorizer(inputCol='list_of_words', outputCol='raw_features')
    cvmodel = cv.fit(sdf)
    result_cv = cvmodel.transform(sdf)
    
    # tf & idf counting
    idf = IDF(inputCol='raw_features', outputCol='features')
    idfModel = idf.fit(result_cv)
    result_tfidf = idfModel.transform(result_cv)
    
    # lda
    lda_model = LDA(k=num_topics)
    lda_model.setMaxIter(max_iterations)
    model = lda_model.fit(result_tfidf)
    
    sdf = model.transform(result_tfidf)
    df['topics'] = sdf.select(['topicDistribution']).toPandas()
    
    # get topic words (visualization)
    topicIndices = model.describeTopics(maxTermsPerTopic = wordNumbers)
    topics_final = topicIndices.select('termIndices').rdd.map(
        lambda topic: [i for i in topic.termIndices]
    ).collect()

    res = [
        [cvmodel.vocabulary[idx] for idx in topics_final[i]]
        for i in range(len(topics_final))
    ]

    return df.drop(['reviews'], axis=1), res


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
        StructField('release_date', StringType(), False), 
        StructField('vote_average', FloatType(), False), 
        StructField('vote_count', IntegerType(), False), 
        StructField('genres', ArrayType(StringType()), False), 
        StructField('budget', LongType(), False), 
        StructField('popularity', FloatType(), False), 
        StructField('revenue', LongType(), False), 
        StructField('release_date_int', LongType(), False), 
        StructField('neg', FloatType(), False), 
        StructField('neu', FloatType(), False), 
        StructField('pos', FloatType(), False), 
        StructField('compound', FloatType(), False),
        StructField('topics', StringType(), False)
    ])

    df = pd.read_csv(path)
    df = df.mask(df == 0).fillna(df.min()).astype(
        {'release_date_int': 'int64', 'vote_count': 'int32', 
         'budget': 'int64', 'revenue': 'int64'}
    )

    # rebuild list objects
    df['genres'] = df['genres'].apply(literal_eval)
    df['topic_lst'] = df['topics'].apply(literal_eval)
    
    # expand topic array to columns
    topic_names = ['topic_{}'.format(i) for i in range(num_topics)]
    df[topic_names] = pd.DataFrame(df['topic_lst'].tolist())
    
    df = df.drop('topic_lst', axis=1)
    
    # append schema
    topics_schema = [
        StructField(tpc, FloatType(), False)
        for tpc in topic_names
    ]
    schema = StructType(schema.fields + topics_schema)

    return sqlContext.createDataFrame(df, schema=schema)


def vectorize(df):
    """
    Vectorize string data and assemble the feature column.
    """
    
    # vectorize genres
    cv_genres = CountVectorizer(inputCol='genres', outputCol='genres_vec', minDF=10)
    cvg_model = cv_genres.fit(df)
    df = cvg_model.transform(df)

    # assemble columns
    inputCols=[
        'vote_count', 'genres_vec', 'budget', 'popularity', 'release_date_int', 
        'revenue', 'neg', 'neu', 'pos', 'compound'#, 'topics'
    ] + ['topic_{}'.format(i) for i in range(num_topics)]

    # assemble features
    va_model = VectorAssembler(
        inputCols=inputCols, 
        outputCol='features'
    )
    df = va_model.transform(df)

    return df


def ExtractFeatureImp(featureImp, dataset, featuresCol):
    """
    Extract feature importance.
    Reference: https://www.timlrx.com/blog/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator
    """

    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))


def validate(data_path=pp_path):
    """
    Train a model and validate it. 
    """

    df = load_data(data_path)
    df = vectorize(df)

    # split for train and validation
    print('Total samples: {}'.format(df.count()))
    train_df, test_df = df.randomSplit([0.8, 0.2])

    # split by score
    N = train_df.count()

    # define model
#    glr = GeneralizedLinearRegression(
#        labelCol='vote_average', 
#        family='gaussian', link='identity', 
#        weightCol='vote_count', 
#        maxIter=10000, regParam=0.05
#    )

    model = RandomForestRegressor(
        labelCol='vote_average', 
        numTrees=30, maxDepth=2, 
        featureSubsetStrategy='all'
    )

    # fit on training set
    print('Train data size: {}'.format(N))
    model = model.fit(train_df)

    # train summary
    #summary = model.summary
    #print("\tCoefficient Standard Errors: " + str(summary.coefficientStandardErrors))
    #print(len(summary.coefficientStandardErrors))
    #print("\tT Values: " + str(summary.tValues))
    #print("\tP Values: " + str(summary.pValues))
    #print("\tDispersion: " + str(summary.dispersion))
    #print("\tNull Deviance: " + str(summary.nullDeviance))
    #print("\tResidual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    #print("\tDeviance: " + str(summary.deviance))
    #print("\tResidual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    #print("\tAIC: " + str(summary.aic))

    # evaluate on test set
    print('Test data size: {}'.format(test_df.count()))
    pred = model.transform(test_df)
    evl = RegressionEvaluator(labelCol='vote_average')
    rmse = evl.evaluate(pred)
    print('Evaluation RMSE: {}'.format(rmse))

    # extract feature importance
    feat = ExtractFeatureImp(model.featureImportances, pred, 'features')

    # save test results
    pred.select(
        ['title', 'prediction', 'vote_average', 'popularity', 
         'release_date', 'vote_count', 'budget', 'revenue', 'neg', 'topics']
    ).toPandas().to_csv('./data/predicts.csv', index=False)
    print('Results saved to ./data/predicts.csv')

    return feat

