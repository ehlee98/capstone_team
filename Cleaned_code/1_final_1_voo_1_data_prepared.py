
import yfinance as yf
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)

start_date = "2022-03-01"
end_date = "2024-11-22"

def fetch_and_process_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.strftime('%Y-%m-%d')
    df['Adj_Closed_price'] = df['Adj Close']
    df['Closed_price'] = df['Close']
    df['Open_price'] = df['Open']
    df['Log_Return'] = np.log(df['Adj_Closed_price'] / df['Adj_Closed_price'].shift(1))
    df['Adj_Open_price'] = df['Open_price'] * (df['Adj_Closed_price'] / df['Closed_price'])
    # only use Next_day_Adj_Open_price for measurement, drop it for model building to prevent data leakage
    df['Next_day_Adj_Open_price'] = df['Adj_Open_price'].shift(-1)
    df_result = df[['Adj_Closed_price','Log_Return','Next_day_Adj_Open_price']].copy()
    df_result = df_result.reset_index().rename(columns={'index': 'Date'})
    return df_result

voo_stock_data=fetch_and_process_stock_data("VOO", start_date, end_date)


def add_technical_indicators(df):
    # # Moving Averages based on Adjusted Close
    df['ma_10'] = df['Adj_Closed_price'].rolling(window=10).mean()
    df['ma_30'] = df['Adj_Closed_price'].rolling(window=30).mean()
    # # Exponential Moving Average based on Adjusted Close
    df['ema_10'] = df['Adj_Closed_price'].ewm(span=10, adjust=False).mean()
    # # Volatility - Standard Deviation of log returns
    df['volatility_10'] = df['Log_Return'].rolling(window=10).std()
    # # Relative Strength Index (RSI) based on Adjusted Close
    delta = df['Adj_Closed_price'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # # Fill NaN values with previous values or a constant if needed
    df = df.ffill().fillna(0)
    return df

voo_stock_data = add_technical_indicators(voo_stock_data)

voo_prepared_data=voo_stock_data

voo_prepared_data.to_csv("voo_prepared_data.csv", index=False)



