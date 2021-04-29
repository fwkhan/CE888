'''This file deals with the pre-processing of the input data.
Using Pandas, numpy and sklearn library for only pre-processing
of the data.'''
import pandas as pd
import requests
from consts import *

'''This class performs  the pre-processing of the data.
   Tasks performed:
   * Reading data stored on the disk in .csv format
   * Splits the read data into train data set and stores the remaining data
   * 70% data is the traininig data and 30% is stored.
   * Remaining 30% stored data is further split in 15% test and 15% validation data set'''


class preprocessing:
    def __init__(self):
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None

    def get_task_link(self, classification_task):
        if classification_task == "sentiment":
            self.train_x = SENTIMENT_TRAIN_TEXT
            self.train_y = SENTIMENT_TRAIN_LABEL

            self.val_x =   SENTIMENT_VALIDATION_TEXT
            self.val_y =   SENTIMENT_VALIDATION_LABEL

            self.test_x =   SENTIMENT_TEST_TEXT
            self.test_y =  SENTIMENT_TEST_LABEL

        if classification_task == "hate":
            self.train_x = HATE_TRAIN_TEXT
            self.train_y = HATE_TRAIN_LABEL

            self.val_x = HATE_VALIDATION_TEXT
            self.val_y = HATE_VALIDATION_LABEL

            self.test_x = HATE_TEST_TEXT
            self.test_y = HATE_TEST_LABEL

        if classification_task == "offensive":
            self.train_x = OFFENSE_TRAIN_TEXT
            self.train_y = OFFENSE_TRAIN_LABEL

            self.val_x = OFFENSE_VALIDATION_TEXT
            self.val_y = OFFENSE_VALIDATION_LABEL

            self.test_x = OFFENSE_TEST_TEXT
            self.test_y = OFFENSE_TEST_LABEL


    # Wrapper to convert text data to pandas Dataframe
    def txt_to_df(self, data, label, classification_task):
        tweet = []
        sentiments = []
        for sentence in data.split('\n'):
            tweet.append(sentence)
        for sentiment in label.split('\n'):
            try:
                sentiments.append(int(sentiment))
            except ValueError:
                pass
        df = pd.DataFrame(tweet[:-1], columns=['tweet'])
        df['label'] = sentiments
        return df
    '''
    In this part, text files are read from github, converted to pandsas dataframe and then processing is done to get rid
     of noise in the data. All the special characters are removed, words are lower-cased, 
     all the words whose length is less than 2 are filtered, getting rid of 'user' from texts and 
     calcualting the length of each tweet and storing it in dataframe.
    '''

    def preprocess(self,df):
        ignore_words = ['user', 'st']
        df['processed_tweets'] = df['tweet'].replace('[^a-zA-Z]', ' ', regex=True,
                                                     inplace=False)
        df['processed_tweets'] = df['processed_tweets'].apply(lambda x: [w.lower() for w in x.split()])
        df['processed_tweets'] = df['processed_tweets'].apply(
            lambda tweet: ' '.join([word for word in tweet if len(word) > 2]))
        df['processed_tweets'] = df['processed_tweets'].apply(
            lambda x: ' '.join([word for word in x.split() if not word in ignore_words]))
        df["sentence_length"] = df.tweet.apply(lambda x: len(str(x).split()))
        return df

    def prepare_dataset(self, classification_task):
        # Reading Train, Vvalidation & Test data from tweeteval Github Repo.
        self.get_task_link(classification_task)

        train_tweets_txt = requests.get(self.train_x).text
        train_labels_txt = requests.get(self.train_y).text

        val_tweets_txt = requests.get(self.val_x).text
        val_labels_txt = requests.get(self.val_y).text

        test_tweets_txt = requests.get(self.test_x).text
        test_labels_txt = requests.get(self.test_y).text

        # Converting text data to pandas Dataframe
        train_df = self.txt_to_df(train_tweets_txt, train_labels_txt, classification_task)
        val_df = self.txt_to_df(val_tweets_txt, val_labels_txt, classification_task)
        test_df = self.txt_to_df(test_tweets_txt, test_labels_txt, classification_task)

        self.train_df = self.preprocess(train_df)
        self.val_df = self.preprocess(val_df)
        self.test_df = self.preprocess(test_df)

        return self.train_df, self.val_df, self.test_df
