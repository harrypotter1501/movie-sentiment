from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
import time

from utils.context import SprkCxt

from config import *


def fetch_data(stream_time, ip, port, out_path):
    try:
        # Spark settings
        conf = SparkConf()
        conf.setMaster('local[1]')
        conf.setAppName("TwitterStreamApp")

        # create spark context with the above configuration
        #sc = SparkContext(conf=conf)
        sc = SprkCxt()
        sc.setLogLevel("ERROR")

        # create sql context, used for saving rdd
        #sql_context = SQLContext(sc)

        # create the Streaming Context from the above spark context with batch interval size 5 seconds
        ssc = StreamingContext(sc, 5)
        # setting a checkpoint to allow RDD recovery
        ssc.checkpoint(cp_path)

        # read data from port 9001
        dataStream = ssc.socketTextStream(ip, port)
        dataStream.pprint()
        dataStream.saveAsTextFiles(out_path, 'txt')

        ssc.start()
        time.sleep(stream_time)
        ssc.stop(stopSparkContext=False, stopGraceFully=True)

    except BaseException as e:
        #print(e)
        pass

