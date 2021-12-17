# configs

# --- streaming ---
# access token
ACCESS_TOKEN = '1440425207190024196-iSOaQja7rMiCMbZwoslmfOe0RX0XPu'
# access secret
ACCESS_SECRET = '2eOScxtfc6aUqfZc4xEDfMTRlxJUIkavB1i0Shfbi6fIt'
# consumer key
CONSUMER_KEY = 'GnCIBr4YWVSFH65ZbOFvXJEpB'
# consumer secret
CONSUMER_SECRET = 'Q15jf7i3M7JcOtevi4EYhCjLblKUrJOdDBKgUAxz5npWVmifVB'

# twtter checkpoint path
cp_path = './data/checkpoint_TwitterApp'
# twitter data output path
output_directory = './data/twitter/movie'

# client IP
IP = 'localhost'
# client port
PORT = 9001

# the tags to track
tags = ['movie']


# --- movies ---
# tmdb api key
api_key = '94b42385a681053cab08a06553dcfa19'
# tmdb language
language = 'en'
# tmdb mode
debug = True

# feature to collect
features_default = [
    'id', 'title', 'release_date', 'vote_average', 'vote_count', 
    'genres', 'budget', 'popularity', 'revenue'
]

# raw data path
path = './data/movies.csv'


# --- preprocess --
# preprocessed data path
pp_path = './data/movies_pp.csv'
# lda num topics
num_topics=10
# lda max iters
max_iterations=50
# lda num of words for each topic
wordNumbers=10


# --- analysis ---
# predicted results path
pred_path = './data/predicts.csv'
topic_path = './data/topics.txt'

