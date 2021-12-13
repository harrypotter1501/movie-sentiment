# context

import pyspark

class SprkCxt:

    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = pyspark.SparkContext()
        return cls.instance

