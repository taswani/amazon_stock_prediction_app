import data_pipeline as dp
from textblob import TextBlob
import pandas as pd
import re
from text_classification import *
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

result_df = sentiment_analysis(result_df, processed_features)

# Necessary function for calculating the r_squared value in neural networks
def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

class DataPreparation:
    '''
    A class to help set up the data for use in machine learning models and neural networks.
    Functions set up to initiate the feature selection, split data in accordance to time series validation,
    and min max scaling in order to normalize the data.
    '''

    def __init__(self, result_df):
        self.result_df = result_df
        self.X = result_df[['Open', 'High', 'Low', 'Average Polarity', 'Polarity']]
        self.y = result_df[['Close']]

    def time_series_split(self, n):
        tscv = TimeSeriesSplit(n_splits=n)
        for train_index, test_index in tscv.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
        return X_train, X_test, y_train, y_test

    def min_max_scaling(self, X_train, X_test, y_train, y_test, only_scaler=False):
        # Scaling all values for a normalized input and output
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)
        y_train = min_max_scaler.fit_transform(y_train)
        y_test = min_max_scaler.transform(y_test)
        if only_scaler is True:
            return min_max_scaler
        else:
            return min_max_scaler, X_train, X_test, y_train, y_test


class Query:
    '''
    Class is set up to take a query and translate the data to be accepted to machine learning and deep learning models
    for the sake of predictions.
    Utilized for taking user data from Flask framework in order to create predictions.
    '''
    def __init__(self, open, high, low, headline, result_df):
        self.open = open
        self.high = high
        self.low = low
        self.headline = headline
        self.average_polarity = result_df['Average Polarity'].iloc[-1]

    def convert_data(self):
        corpus = self.headline
        # Remove all the special characters
        processed_feature = re.sub(r'[^a-zA-Z\s]', '', corpus, re.I|re.A)
        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)
        # Converting to Lowercase
        processed_feature = processed_feature.lower().strip()
        processed_features.append(processed_feature)
        sentence = TextBlob(processed_feature)
        polarity = sentence.sentiment.polarity
        data = [self.open, self.high, self.low, self.average_polarity, polarity]
        return data

# TODO: At this point I would pass it to the models in order to get a prediction.
# Not really concerned about the date as the prediction is assumed to be in the past. Might want to start with a few features first.
