# spark streaming
"""
File modified from starter code in Homework 1.
"""

# modules
from pyspark import SparkConf
from pyspark.streaming import StreamingContext
import time

# context manager
from utils.context import SprkCxt

# configs
from config import *


def fetch_data(stream_time, ip, port, out_path):
    """
    Fetch streaming data for a given time.
    """

    try:
        # Spark settings
        conf = SparkConf()
        conf.setMaster('local[1]')
        conf.setAppName("TwitterStreamApp")

        # create spark context with the above configuration
        sc = SprkCxt()
        sc.setLogLevel("ERROR")

        # create the Streaming Context from the above spark context with batch interval size 5 seconds
        ssc = StreamingContext(sc, 5)
        # setting a checkpoint to allow RDD recovery
        ssc.checkpoint(cp_path)

        # read data
        dataStream = ssc.socketTextStream(ip, port)
        dataStream.pprint()
        dataStream.saveAsTextFiles(out_path, 'txt')

        ssc.start()
        time.sleep(stream_time)
        ssc.stop(stopSparkContext=False, stopGraceFully=True)

    except BaseException as e:
        #print(e)
        pass

