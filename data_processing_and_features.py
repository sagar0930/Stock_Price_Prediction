import pandas as pd
from textblob import TextBlob

def news_sentiment(df_news):
    df_news['headline'] = df_news['headline'].astype(str)
    df_news['summary'] = df_news['summary'].astype(str)
    
    def calculate_sentiment_polarity(text):
        return TextBlob(text).sentiment.polarity    

    df_news['headline_sentiment'] = df_news['headline'].apply(calculate_sentiment_polarity)
    df_news['summary_sentiment'] = df_news['summary'].apply(calculate_sentiment_polarity)
    return df_news

def merging_df(df_news,df_price):
    df_news["date"] = pd.to_datetime(df_news["datetime"], unit="s").dt.date
    df_news['Date'] = pd.to_datetime(df_news['date'])
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    merged_df = pd.merge(df_price, df_news, on = 'Date', how = 'left' )
    return merged_df

def features_handling(merged_df):
    column_to_drop = ['category','headline','id','image','related','source','summary','url','date','datetime','Adj Close']
    merged_df.drop(columns=column_to_drop, inplace = True)
    return merged_df

#we have only 14 values having null values for sentiment because no news on those day hence replace with 0
def missing_value_imputation(merged_df):
    merged_df.fillna(0.00, inplace = True)
    return merged_df

# As we have multiple news for same day and have multiple sentimenst hence takenday wise mean 
def final_data(merged_df):
    final_df = merged_df.groupby('Date').mean().reset_index()
    #final_df['Date'] = pd.to_datetime(final_df['Date'], format='%d/%m/%Y')
    final_df['Date'] = final_df['Date'].apply(lambda x: x.to_pydatetime())
    return final_df


