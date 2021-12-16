# twitter

# modules
from utils.twitterHTTPClient import twitter_client

from utils.sparkStreaming import fetch_data
from threading import Thread

# configs
from config import *


def get_streaming(tags, time, IP=IP, PORT=PORT):
    """
    Get streaming with a newly created client thread.
    The thread will stop automatically when time elapsed and free the resources.
    """

    # client thread
    client = twitter_client(IP, PORT)
    thread = Thread(target=client.run_client, args=(tags,))
    thread.start()

    fetch_data(time, IP, PORT, output_directory)
    client.stop_client()

