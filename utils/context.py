# context

# import module
import pyspark

class SprkCxt:
    """
    Managing SparkContext. 
    Use singleton to prevent problems caused by multiple creation across different functions. 
    """

    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = pyspark.SparkContext()
        return cls.instance

