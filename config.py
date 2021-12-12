# configs

# streaming
from pyspark.sql.context import SQLContext


ACCESS_TOKEN = '1440425207190024196-iSOaQja7rMiCMbZwoslmfOe0RX0XPu'
ACCESS_SECRET = '2eOScxtfc6aUqfZc4xEDfMTRlxJUIkavB1i0Shfbi6fIt'
CONSUMER_KEY = 'GnCIBr4YWVSFH65ZbOFvXJEpB'
CONSUMER_SECRET = 'Q15jf7i3M7JcOtevi4EYhCjLblKUrJOdDBKgUAxz5npWVmifVB'

cp_path = './data/checkpoint_TwitterApp'
output_directory = './data/twitter/movie'

IP = 'localhost'
PORT = 9001

# the tags to track
tags = ['movie']

# movies
api_key = '94b42385a681053cab08a06553dcfa19'
language = 'en'
debug = True
features_default = [
    'id', 'title', 'release_date', 'vote_average', 'vote_count', 
    'genres', 'budget', 'popularity', 'revenue'
]
path = './data/movies.csv'

# spark

