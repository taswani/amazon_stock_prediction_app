import pandas as pd
import numpy as np
import datetime as dt
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from dask import dataframe as dd
import re

# In case you need data
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

price_csv = "./data_modules/data_csv/AMZN.csv"
headline_csv = "./data_modules/data_csv/combined_amazon_date_data.csv"

#Function for datetime conversion
def datetime_conversion(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    return df

# Function that takes in the file path of the data associated with the price and headlines
def prepare_data(price_csv, headline_csv):
    # read in data
    amazon_df = pd.read_csv(headline_csv, index_col=False)
    amzn_df = pd.read_csv(price_csv, index_col=False)
    # adjust datetime
    datetime_conversion(amazon_df)
    datetime_conversion(amzn_df)
    # merging
    amzn_df = amzn_df.set_index('Date').join(amazon_df.set_index('Date'))
    result_df = amzn_df
    # dates being sorted for forward fill
    result_df = result_df.sort_values(by='Date')
    # cover null values in data
    result_df = result_df.fillna(method='ffill')
    result_df = result_df.fillna(method='bfill')
    # Changing to Dask dataframe in order to parallelize the data wrangling (use case for large datasets)
    result_df = dd.from_pandas(result_df, npartitions=1)
    # dropping any mention of anything other than the company amazon
    # work in progress as I see other keywords
    to_drop = ['rainforest', 'forest', 'Brazil', 'river', 'jungle', 'River', 'pilots', 'gangs', 'drugs', 'OkCupid', 'dating']
    result_df = result_df[~result_df['Headlines'].str.contains('|'.join(to_drop))]
    # combining duplicates by date
    result_df = result_df.groupby(['Date', 'Open', 'High', 'Low', 'Close'])['Headlines'].apply('// '.join, meta=pd.Series(dtype='str', name='Headlines')).reset_index()
    # Changing Dask dataframe to Pandas for reindexing
    result_df = result_df.compute()
    # setting index as dates, and imputing missing dates
    # Every value is forward filled as a necessity to keep everything in sync
    result_df = result_df.set_index('Date')
    idx = pd.date_range('2016-06-30', '2019-08-11')
    result_df = result_df.reindex(idx)
    result_df = result_df.fillna(method='ffill')
    # Adding a rolling window mean
    # TODO: add minimum mean and maximum mean as potential features
    result_df['Average Mean'] = result_df[['Close']].rolling(window = 100).mean()
    result_df = result_df.fillna(method='bfill')
    # Adding a differential column
    result_df['Differential'] = result_df['High'].values - result_df['Low'].values
    return result_df

def vectorization(result_df):
    #Creation of corpus from all the headlines
    corpus = []
    for line in result_df['Headlines']:
        corpus.append(line)
    #preprocess text for vectorization
    processed_features = []
    for sentence in range(0, len(corpus)):
        # Remove all the special characters
        processed_feature = re.sub(r'[^a-zA-Z\s]', '', corpus[sentence], re.I|re.A)
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
    #Main part of the data pipeline
    # #Initializing Vectorizer
    # vectorizer = TfidfVectorizer(stop_words = stopwords.words('english'))
    # #Transforming words to vectors
    # processed_features = vectorizer.fit_transform(processed_features).toarray()
    # corpus_df = pd.DataFrame(processed_features, columns = vectorizer.get_feature_names())
    return processed_features

#TODO: fitting model on training and testing set as part of the data pipeline
# Data should already preprocessed, mainly acting on training and testing data post wrangling
# Create other functions to help validate the data in different ways
# look into result_df['Headlines'].str.lower()

# TODO: Look at unit 17 for refactoring tips
