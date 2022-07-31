import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

login = pd.read_csv("../TwitterLogin.csv")
bearerToken = login['key'][0]
consumerKey = login['key'][1]
consumerSecret = login['key'][2]
accessToken = login['key'][3]
accessTokenSecret = login['key'][4]

client = tweepy.Client(bearer_token=bearerToken, consumer_key=consumerKey, consumer_secret=consumerSecret, access_token=accessToken, access_token_secret=accessTokenSecret)

tweets = client.search_recent_tweets(query="#covid19 lang:en -is:retweet",  max_results=100)


df = pd.DataFrame([tweet.text for tweet in tweets.data], columns=["Tweets"])

df.head()

def cleanTxt(text):
	text = re.sub(r'@\w+','', text)
	text = re.sub(r'#\w+','', text)
	text = re.sub(r'http\S+','',text)
	return text

df['Tweets'] =  df['Tweets'].apply(cleanTxt)

print(df['Tweets'])

def getSubjectivity(text):
	return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
	return TextBlob(text).sentiment.polarity

df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)


def getAnalysis(score):
	if score < 0:
		return 'Negative'
	elif score == 0:
		return 'Neutral'
	else:
		return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)

print(df['Subjectivity'])
print(df['Polarity'])
print(df['Analysis'])

j=1
sortedDF= df.sort_values(by=['Polarity'])
for i in range(0, sortedDF.shape[0]):
	if sortedDF['Analysis'][i] == 'Positive':
		print(str(j)+') '+sortedDF['Tweets'][i])
		j+=1

j=1
sortedDF2 = df.sort_values(by=['Polarity'], ascending=False)
for i in range(0, sortedDF2.shape[0]):
	if sortedDF2['Analysis'][i] == 'Negative':
		print(str(j)+') '+sortedDF2['Tweets'][i])
		j+=1

plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
	plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Blue')
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()