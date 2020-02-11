import data_pipeline as dp
from textblob import TextBlob

result_df = dp.prepare_data(dp.price_csv, dp.headline_csv)
processed_features = dp.vectorization(result_df)

# Look to use pre-trained model, aggregrate the predictions and append to result_df

def sentiment_analysis(result_df, processed_features):
    # List for taking in the sentiments associated with the processed features
    sentiments = []
    polarity = []
    # Using a pre-trained model to analyze sentiment for each headline
    for feature in processed_features:
        sentence = TextBlob(feature)
        polarity.append(sentence.sentiment.polarity)
        if sentence.sentiment.polarity > 0:
            sentiments.append(1)
        else:
            sentiments.append(0)
    # Adding a new column in the dataframe and returning it
    result_df['Sentiment'] = sentiments
    result_df['Polarity'] = polarity
    result_df['Average Polarity'] = result_df[['Polarity']].rolling(window = 100).mean()
    result_df = result_df.fillna(method='bfill')
    # pushing df to a csv
    result_df.to_csv("final_amazon.csv")
    return result_df
