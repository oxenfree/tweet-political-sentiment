import copy

from config import consumer_key, consumer_secret, access_token_secret, access_token
import json
import re
import requests
import time
import tweepy
import pandas as pd
from textblob import TextBlob
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

sentiment_url = 'http://text-processing.com/api/sentiment/'
stemming_url = 'http://text-processing.com/api/stem/'


class TweetHandler:

    def __init__(self):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

    def get_tweet_attributes_map(self, user_timeline, limit=1):
        tweet_map = {}
        count = 0
        cursor = tweepy.Cursor(
            self.api.user_timeline, id=user_timeline, include_entities=True, lang="en"
        ).items()

        while count < limit:
            try:
                tweet = cursor.next()
                tweet_map[count] = {
                    'date': tweet.created_at,
                    'favorite_count': tweet.favorite_count,
                    'retweet_count': tweet.retweet_count,
                    'text': tweet.text
                }
                count += 1
            except tweepy.TweepError:
                time.sleep(60 * 15)
                continue
            except StopIteration:
                break

        return tweet_map

    @staticmethod
    def clean_tweet_map_texts(tweet_map):

        for i, v in enumerate(tweet_map.items()):
            tweet = tweet_map[i]['text']
            tweet = ' '.join(re.sub("(@)", "", tweet).split())
            tweet = tweet.replace('RT', '')
            tweet_map[i]['text'] = tweet.replace('#', '')

        return tweet_map

    @staticmethod
    def add_nltk_sentiment_map(tweet_map):
        count = 0

        while count < len(tweet_map):
            try:
                data = {"text": tweet_map[count]['text']}
                sentiment_response = requests.post(sentiment_url, data)
                probs = json.loads(sentiment_response.content.decode('utf-8'))['probability']
                tweet_map[count]['nltk_pos'] = float(probs["pos"])
                tweet_map[count]['nltk_neu'] = float(probs["neutral"])
                tweet_map[count]['nltk_neg'] = float(probs["neg"])
                count += 1
            except StopIteration:
                break

        return tweet_map

    @staticmethod
    def add_text_blob_sentiment_map(tweet_map):

        def analyze(tweet_text):
            analysis = TextBlob(tweet_text)
            if analysis.sentiment.polarity > 0:
                return 1
            if analysis.sentiment.polarity == 0:
                return 0

            return -1

        for i, tweet in enumerate(tweet_map.items()):
            tweet_map[i]['blob_sent'] = analyze(tweet_map[i]['text'])

        return tweet_map


def clean_add_sents(frame_in):
    frame = copy.deepcopy(frame_in)
    temp = frame.to_dict(orient='index')
    cleaned_temp = TweetHandler.clean_tweet_map_texts(temp)
    sent_temp = TweetHandler.add_nltk_sentiment_map(cleaned_temp)
    sent_temp = TweetHandler.add_text_blob_sentiment_map(sent_temp)
    frame = pd.DataFrame().from_dict(sent_temp, orient='index')
    return frame


if __name__ == '__main__':
    pa_gov_handles = ['realScottWagner', 'WolfForPA']
    fl_gov_handles = ['AndrewGillum', 'RonDeSantisFL']
    ca_34_handles = ['JimmyGomezCA', 'AhnforCA34']  # Democrats only. Runoff: 6/6/17
    ks_04_handles = ['LauraforKansas', 'JamesThompsonKS']  # Democrats only: 4/11/17
    ga_06_handles = ['karenhandel', 'ossoff']  # General R and D Runoff: 6/26/17
    sc_05_handles = ['RalphNorman', 'Archie4Congress']  # General R and D General: 6/20/17
    mt_all_handles = ['GregForMontana', 'RobQuistforMT']  # General R and D General: 5/25/17
    all_races = [
        pa_gov_handles,
        fl_gov_handles,
        ca_34_handles,
        ks_04_handles,
        ga_06_handles,
        sc_05_handles,
        mt_all_handles
    ]
    handler = TweetHandler()
    print('3, 2, 1 ... contact!')
    tweet_dict = {}
    for filename in all_races:
        for name in filename:
            print(f'Collecting tweets for {name}')
            tweet_dict[name] = handler.get_tweet_attributes_map(name, 500)

    print('Writing pandas out to csv.')
    for key in tweet_dict.keys():
        pd.DataFrame.from_dict(tweet_dict[key], orient='index').to_csv(f'data/{key}.csv', index=False)

    print('Ding! Fries are done.')

'''
################ SETUP #####################
############################################

In the same directory as this script, you'll need
a file called 'config.py'. In that file, you'll 
need to put your twitter authentication credentials. 
Such as:

consumer_key = '_your_twitter_key_'
consumer_secret = '_your_twitter_secret_'
access_token = '_your_token_'
access_token_secret = '_your_token_secret_'

These are imported at the top of this script and are required
for the script to run correctly.

################ USAGE #####################
############################################

1) Pick a search phrase to pull tweets from twitter:

user_timeline = 'twitter' # change this to the search phrase

2)  Set a search limit, the total number of tweets to 
    pull in. Keep in mind twitter has a daily limit on 
    API calls:
 
tweet_limit = 3  # change this to the number of tweets you want

3) Make a new TweetHandler, like this:

handler = TweetHandler()

4) Pull tweets with your search phrase and size limit:

tweets = handler.get_tweet_attributes_map(search_phrase, tweet_limit)

5)  Optional: you can take out the 'RT' and '@' symbols 
    from the tweet texts.
    
clean = handler.clean_tweet_map_texts(tweets)

6) Optional: get the nltk_sentiment scores for each tweet:

sent_map = handler.add_nltk_sentiment_map(clean)

7)  Optional: geth the TextBlob sentiment score for each tweet.
    TextBlob ranks phrases by -1, 0, 1 (-1: negative, 0: neutral, 1: positive)
sent_map = handler.add_text_blob_sentiment_map(sent_map)

8)  Optional: stem the tweet texts. Stemming can change the meaning
    and sentiment of a phrase though.
stems = handler.stem_tweet_texts(cleaned_tweets)

################ DATA ######################
############################################

You will now have a number of tweets with a datetime, text,
device (such as "Android" or "iPhone" or "Web"), and optionally 
you can have stemmed or sentiment analyzed texts for each tweet. 

The tweet dictionary structure is like:
{
    0: {
        date: datetime.datetime(2018, 8, 30, 19, 24, 47),
        'device': 'Twitter for Android',
        'text': "As resident of AndrewGillum's Tallahassee...',
        'nltk_pos': 0.26249691321696916,
        'nltk_neu': 0.8573198090476511,
        'nltk_neg': 0.7375030867830308,
        'blob_sent': -1
    },
    1: {
        'date': datetime.datetime(2018, 8, 30, 19, 24, 6),
        'device': 'Twitter for Android',
        'text': ' ScottPresler: The socialist and democrat ...',
        'nltk_pos': 0.592341302535156,
        'nltk_neu': 0.8948539758194366,
        'nltk_neg': 0.40765869746484396,
        'blob_sent': 0
    }
}

Note: This example shows unstemmed tweets. Stemmed tweets will have the same
structure, but the text will be stemmed.
'''
