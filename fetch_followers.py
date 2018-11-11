from __future__ import division
import tweepy
import time
import os
import statistics 
import os
import twitter
#insert your Twitter keys here
consumer_key ="9fi7SGkNTmJ7aB0mAqk2MnD32"
consumer_secret="rQWLQ5DDiYuRjELtYF521G6LQqQzKQpkR7jjF8tu5PMa7c5g3I"
access_token="907636142987558914-NXtnukVQoL8OqkbCQg0xgMhbx7z48NY"
access_token_secret="NVoOzE69pn17DFW996KAuHsqisyStsRshOREgL004qUjB"



twitter_handle='realDonaldTrump'
if not os.path.exists(twitter_handle):
    os.makedirs(twitter_handle)

auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
fname=twitter_handle+'//followers.txt'

list= open(fname,'w+')

if(api.verify_credentials):
    print ('We successfully logged in')


user = tweepy.Cursor(api.followers, screen_name=twitter_handle).items()

while True:
    try:
        u = next(user)
	print("Getting Followers of "+twitter_handle)
        list.write(u.screen_name +' \n')

    except:
	break
        time.sleep(15*60)
        print ('We got a timeout ... Sleeping for 15 minutes')
        u = next(user)
        list.write(u.screen_name +' \n')
list.close()




	







