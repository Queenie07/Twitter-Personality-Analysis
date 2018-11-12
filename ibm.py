
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from datetime import date, datetime, timedelta
from LIWCutils import LIWCdict
import statistics
import arff
import fnmatch
import pickle

import re
import sys
import itertools
import collections

import os
import operator
import requests
import json
import http.client as http_client
import twitter
from tweepy import OAuthHandler
from tweepy import API
from watson_developer_cloud import PersonalityInsightsV3 as PersonalityInsights
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import csv
import fnmatch



reload(sys)  
sys.setdefaultencoding('utf8')



def get_twitter_client():
  access_key = "907636142987558914-NXtnukVQoL8OqkbCQg0xgMhbx7z48NY"
  access_secret = "NVoOzE69pn17DFW996KAuHsqisyStsRshOREgL004qUjB"
  consumer_key = "9fi7SGkNTmJ7aB0mAqk2MnD32"
  consumer_secret = "rQWLQ5DDiYuRjELtYF521G6LQqQzKQpkR7jjF8tu5PMa7c5g3I"

  twitter_api = twitter.Api(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token_key=access_key, access_token_secret=access_secret)
  return twitter_api

def get_pi_client():
  username= "e1c4f5cf-36bf-4da3-83db-b80c377cafa4",
  password= "OorUbo1mtSBY"
  personality_insights = PersonalityInsights(version='2018-10-25',username='e1c4f5cf-36bf-4da3-83db-b80c377cafa4', password='OorUbo1mtSBY')

  return personality_insights

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def analyze(handle):
  # Retrieve data from Twitter.
  #statuses = get_twitter_client().GetUserTimeline(screen_name=handle, count=200, include_rts=False)
  try: 
   max_id = None
   statuses = []
   #for x in range(0, 16):  # Pulls max number of tweets from an account
	#    if x == 0:
   statuses_portion = get_twitter_client().GetUserTimeline(screen_name=handle,
	                                                       count=200,
	                                                       include_rts=False)
#	        status_count = len(statuses_portion)
#	        max_id = statuses_portion[status_count - 1].id   # get id of last tweet and bump below for next tweet set
#	    else:
#	        statuses_portion = get_twitter_client().GetUserTimeline(screen_name=handle,
#	                                                       count=200,
#	                                                       max_id=max_id,
#	                                                       include_rts=False)
#	        status_count = len(statuses_portion)
#	        max_id = statuses_portion[status_count - 1].id   # get id of last tweet and bump below for next tweet set
   for status in statuses_portion:
	        statuses.append(status)
  # This variable saves tweets.
  # Convert to bytes first.
   text = b"" 

   nwords=0
   ltweets=0
   lenlist=[]
   num=0

   m=0
   mentions=[]
   hashl=[]

   for status in statuses:
    #remove emoji
    status.text=remove_emoji(status.text)
    if (status.lang == 'en'):

      text += status.text.encode('utf-8')
      temp=len(status.text.encode('utf-8'))
      ltweets=temp+ltweets
      num=num+1
      print("Getting Tweets of "+handle)



      lenlist.append(temp)
      mentions.append(status.text.encode('utf-8').count("@"))
      hashl.append(status.text.encode('utf-8').count("#"))
      m=m+1
      if temp>nwords:
	nwords=temp


  except UnicodeDecodeError:
    print("")


  comma=text.count(',')
  colon=text.count(':')
  quest=text.count('?')
  exclam=text.count('!')
  paran=text.count('(')
  hasht=text.count('#')
  links=text.count("http")

  fname=handle+'//followers.txt'
  with open(fname) as f:
    content = f.readlines()
  content = [x.strip() for x in content] 
  nfollow=len(content)

  try:
   STD_Attn=statistics.stdev(mentions)
   Std_HASH=statistics.stdev(hashl)
   STD_len=statistics.stdev(lenlist)
  except statistics.StatisticsError:
   STD_Attn=len(mentions)
   Std_HASH=len(hashl)
   STD_len=len(lenlist)
  HASH_M=hasht/m
  LEN=ltweets/num
  Attn=0
  for m in mentions:
    Attn=Attn+m
  Attn=Attn/nfollow

                 
  fname=handle+"//follower_tweets.txt"
  with open(fname) as f:
      fi = f.readlines()
      h=fi.count("#")
      l=len(fi)
      m = 1
      while fi:
       fi = f.readline()
       m += 1


  FFhash=h/m
  FFlen=l/m
  
  access_token = "907636142987558914-NXtnukVQoL8OqkbCQg0xgMhbx7z48NY"
  access_token_secret = "NVoOzE69pn17DFW996KAuHsqisyStsRshOREgL004qUjB"
  consumer_key = "9fi7SGkNTmJ7aB0mAqk2MnD32"
  consumer_secret = "rQWLQ5DDiYuRjELtYF521G6LQqQzKQpkR7jjF8tu5PMa7c5g3I"
  auth = OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  auth_api = API(auth)

  item = auth_api.get_user(handle)
  FR=item.friends_count
  FO=item.followers_count
  account_created_date = item.created_at
  delta = datetime.utcnow() - account_created_date
  account_age = delta.days
  if account_age > 0:
     avg=item.statuses_count/account_age






  text=text.lower()
  print(text)


  tokens = word_tokenize(text)
  doc = []
  stop_words = set(stopwords.words('english'))
  for w in tokens:
   try:
    if w not in stop_words:
        doc.append(w)

   except UnicodeDecodeError:
    print("")

  print("After tokenization")
  print(tokens)

  print("After stopword removal")
  print(doc)


  ps = PorterStemmer()
  l=[]
  for w in doc:
   try:
    l.append(ps.stem(w))
   except UnicodeDecodeError:
    print("")
   print("stemming result")
   print(l)

  doc=[]
  fname=handle+"//follower_tweets.txt"
  list=open(fname,'r')
  f=list.read()
  f=str(f)
  prop=0
  for w in l:#removing special chars
   try:
    if (w.isalnum()):
        doc.append(w)
    if w in f:
	prop=prop+1





   except UnicodeDecodeError:
    print("")

  prop=prop/len(doc)

 # print(doc)

  catfile = 'liwccat2007.txt'
  dicfile = 'liwcdic2007.dic'
  liwc = LIWCdict(catfile, dicfile)
  le=0
  text1=text
  print(doc)

  for w in text1:
  	#print(w)
	le=le+1
  #print(le)
    # print all markers
  print('All 64 markers: ')
  print(liwc.get_markers(sort='A'))
  count=[]
  score=[]
  m=liwc.get_markers(sort='A')
  for markers in liwc.get_markers(sort='A'):
	c=0
	for lex in liwc.marker_lexemes(markers,include_wc=False):
	     try:
		if lex in text:
			c=c+1
             except UnicodeDecodeError:
		print("")

	count.append(c)
	score.append(c/le)
  #print(count)
	
  dictionary = dict(zip(m, score))
  #print(dictionary)
  score=dictionary
  pscore=[]
  w1=[]
  openn=(0.396 * score['article'] - 0.15 *score['home'] -0.239 * score['bio'] -0.299*score['body']+0.2388*score['quant']+0.426*score['work']+0.251 * score['humans']+0.264 * score['cause']+ 0.347 * score['certain']-exclam/len(text1)*0.295+0.298*hasht/len(text1))
  pscore.append(openn)
  consc=0.252*score['you']-0.284*score['auxverb']-0.286*score['future']-0.374*score['negate']-0.268*score['negemo']-0.253*score['sad']-0.244*score['cogmech']-0.292*score['discrep']-0.236*score['feel']+0.33*score['work']-0.332*score['death']-0.272*score['filler']-0.24*comma/len(text1)+0.322*colon/len(text1)+0.26*exclam/len(text1)+0.256*links/len(text1)
  pscore.append(consc)
  extro=(0.262 * score['social'] + 0.338 *score['family'] -0.277 * score['health'] + quest*0.263/len(text1) + nwords*0.285/len(text1))
  pscore.append(extro)
  agree=(0.364 * score['you'] -0.258 * score['cause']+0.247 *score['ingest']-0.24 *score['achieve']-0.259 * score['money']+0.206*score['bio']+0.153*colon/len(text1)+0.08*comma/len(text1))
  pscore.append(agree)
  neuro=0.335*score['hear']+0.244*score['feel']+0.383*score['relig']-0.217*hasht/len(text1)+0.26*exclam/len(text1)
  pscore.append(neuro)
  heads=['openness','consci','extraversion','agreeablity','neuro']
  i=0
  nscore=[]
  for p in pscore:
	norm=round((6*(p-min(pscore))/(max(pscore)-min(pscore)))+1)
        if(p==min(pscore)):
		norm=norm+1
        if(p==max(pscore)):
		norm=norm-1
        nscore.append(norm)
  pscore=nscore
  print(nscore)
  filew=[]
  filew.append(handle)
  filew.append(text)
  for n in nscore:
   if n<2.5:
	   filew.append("Low")
   else:
     if n>4.5:
	   filew.append("High")
     else:
	   filew.append("Medium")
  with open('data1.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(filew)







  pscore2=[]
  openn2=-0.263*account_age+0.315*HASH_M+0.256*Std_HASH+0.293*LEN
  pscore2.append(openn2)
  consc2=-0.264*FFhash-0.315*FFlen
  pscore2.append(consc)
  extro2=0.302*LEN-0.278*prop+0.207*STD_Attn
  pscore2.append(extro2)
  agree2=0.259*HASH_M+0.310*Std_HASH
  pscore2.append(agree2)
  neuro2=0.302*FO-0.273*account_age+0.315*STD_len-0.310*Attn
  pscore2.append(neuro2)
  heads=['openness','extraversion','agreeablity','consci','neuro']
  i=0
  nscore=[]
  for p in pscore2:
	norm=round((6*(p-min(pscore2))/(max(pscore2)-min(pscore2)))+1)
        if(p==min(pscore2)):
		norm=norm+1
        if(p==max(pscore2)):
		norm=norm-1
        nscore.append(norm)
  pscore2=nscore
  print(nscore)

  filew=[]
  filew.append(handle)
  filew.append(text)
  for n in nscore:
   if n<2.5:
	   filew.append("Low")
   else:
     if n>4.5:
	   filew.append("High")
     else:
	   filew.append("Medium")
  with open('data2.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(filew)



  
  pi_result = get_pi_client().profile(text,content_type = 'text/plain;charset=utf-8',raw_scores = True,consumption_preferences = True)
  pi=[]
  nscore=[]
  for trait in pi_result.get_result()['personality']:
     pi.append(trait['raw_score'])
  for p in pi:
	norm=round((6*(p-min(pi))/(max(pi)-min(pi)))+1)
        if(p==min(pi)):
		norm=norm+1
        if(p==max(pi)):
		norm=norm-1
        nscore.append(norm)

  filew=[]
  filew.append(handle)
  filew.append(text)
  for n in nscore:
   if n<2.5:
	   filew.append("Low")
   else:
     if n>4.5:
	   filew.append("High")
     else:
	   filew.append("Medium")
  with open('data3.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(filew)




user_handle = "realDonaldTrump"


analyze(user_handle)






