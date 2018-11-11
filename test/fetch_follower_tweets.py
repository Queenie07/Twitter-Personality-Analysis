import twitter
import tweepy
from tweepy import OAuthHandler
from tweepy import API

consumer_key ="9fi7SGkNTmJ7aB0mAqk2MnD32"
consumer_secret="rQWLQ5DDiYuRjELtYF521G6LQqQzKQpkR7jjF8tu5PMa7c5g3I"
access_token="907636142987558914-NXtnukVQoL8OqkbCQg0xgMhbx7z48NY"
access_token_secret="NVoOzE69pn17DFW996KAuHsqisyStsRshOREgL004qUjB"
auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

twitter_handle='BookQuillRant'
fname=twitter_handle+'//followers.txt'

with open(fname) as f:
    content = f.readlines()
content = [x.strip() for x in content] 



for follow in content:
	statuses = api.user_timeline(screen_name=follow, count=200, include_rts=False)
	text = b"" 
	fname=twitter_handle+'//follower_tweets.txt'
	list=open(fname,'w+')
	for status in statuses:
	    if (status.lang == 'en'):
      		text += status.text.encode('utf-8')
	print("Getting Tweets of "+follow)
        list.write(text +' \n')

	list.close()


	







