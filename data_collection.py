import finnhub
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Setup client
finnhub_client = finnhub.Client(api_key='comddihr01qqra7h1bcgcomddihr01qqra7h1bd0')

def get_news_data():
    # Initialize an empty DataFrame
    df_news = pd.DataFrame()

    # Initial start date
    start_date = '2023-04-01'
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Loop for one year
    for _ in range(73):        
        start_date_obj += timedelta(days=1)
        start_date_new = start_date_obj.strftime('%Y-%m-%d')          
        start_date_obj += timedelta(days=4)        
        end_date = start_date_obj.strftime('%Y-%m-%d')
        
        # Fetch company news from Finnhub API
        res = finnhub_client.company_news('AAPL', _from=start_date_new, to=end_date)
        
        # Convert response to DataFrame and concatenate
        df1 = pd.DataFrame(res)
        df_news = pd.concat([df_news, df1])
    return df_news

def get_stock_price():
    # Define the stock symbol (AAPL for Apple Inc.)
    symbol = 'AAPL'
    
    # Fetch stock data from Yahoo Finance
    df_price = yf.download(symbol, start='2023-04-01', end='2024-03-31')
    df_price.reset_index(inplace=True)

    return df_price

# Call the functions
df_news = get_news_data()
df_price = get_stock_price()

# Save df_news to CSV file
df_news.to_csv('news_data.csv', index=False)

# Save df_price to CSV file
df_price.to_csv('stock_price.csv', index=False)