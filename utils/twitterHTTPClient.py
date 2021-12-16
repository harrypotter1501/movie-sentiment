# twitter client
"""
File modified from starter code in Homework 1.
"""

# modules
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json

# configs
from config import *


class TweetsListener(StreamListener):
    """
    Tweets listener object.
    """

    def __init__(self, csocket):
        self.client_socket = csocket

    def on_data(self, data):
        try:
            msg = json.loads(data)
            self.client_socket.send(msg['text'].encode('utf-8'))
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            return False

    def on_error(self, status):
        print(status)
        return False


def sendData(c_socket, tags):
    """
    Send data to socket.
    """

    auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    twitter_stream = Stream(auth, TweetsListener(c_socket))
    twitter_stream.filter(track=tags, languages=['en'])


class twitter_client:
    """
    Twitter client object.
    """

    def __init__(self, TCP_IP, TCP_PORT):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((TCP_IP, TCP_PORT))

    def run_client(self, tags):
        """
        Run client to recieve streaming.
        """
        try:
            self.s.listen(1)
            while True:
                print("Waiting for TCP connection...")
                conn, _ = self.s.accept()
                print("Connected... Starting getting tweets.")
                sendData(conn, tags)
                conn.close()
        except KeyboardInterrupt:
            self.stop_client()

    def stop_client(self):
        """close the socket connection to free resources"""
        self.s.close()


if __name__ == '__main__':
    # development testing
    client = twitter_client('localhost', 9001)
    client.run_client(['data'])

