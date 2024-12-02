
import yfinance as yf
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)

start_date = "2022-03-01"
end_date = "2024-11-22"
ticker='CRM'
ticker_lowercase = ticker.lower()

# price data
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

stock_data=fetch_and_process_stock_data(ticker, start_date, end_date)

# technical data
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

stock_data = add_technical_indicators(stock_data)

# economic_data
economic_data = pd.read_csv(f"data/economic_data_{ticker}.csv")


def calculate_mom_change(df, variable_name, ratio=False):
    """
    Calculate month-over-month change for a given variable in a daily DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with a DateTime index and daily values for the specified variable.
    - variable_name (str): The name of the variable/column to calculate MoM change for.
    - ratio (bool): If True, calculates month-over-month percentage change; if False, calculates raw difference.

    Returns:
    - pd.DataFrame: Updated DataFrame with a new column for month-over-month change.
    """
    # Ensure date is the index if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # Resample to monthly end frequency using the last available value for each month
    monthly_data = df[variable_name].resample('ME').last().to_frame()

    # Calculate month-over-month change based on ratio flag
    if ratio:
        monthly_data[f'{variable_name}_mom'] = monthly_data[variable_name].diff()
    else:
        monthly_data[f'{variable_name}_mom'] = monthly_data[variable_name].pct_change(fill_method=None) * 100

    # Merge MoM data back to daily data
    df = df.merge(monthly_data[[f'{variable_name}_mom']], left_index=True, right_index=True, how='left')

    # Forward-fill the NaN values in the new column to propagate the monthly value across all days in the month
    df[f'{variable_name}_mom'] = df[f'{variable_name}_mom'].ffill()

    # Reset the index if needed
    df = df.reset_index()

    return df

# For percentage change (MoM as ratio)
economic_data = calculate_mom_change(economic_data, 'interest_rate', ratio=True)
economic_data = calculate_mom_change(economic_data, 'unemployment_rate', ratio=True)
economic_data = calculate_mom_change(economic_data, 'gdp_growth', ratio=True)

# For raw difference (absolute MoM change)
economic_data = calculate_mom_change(economic_data, 'business_investment', ratio=False)
economic_data = calculate_mom_change(economic_data, 'software_spending', ratio=False)
economic_data = calculate_mom_change(economic_data, 'corporate_earnings', ratio=False)
economic_data = calculate_mom_change(economic_data, 'consumer_sentiment', ratio=False)
economic_data = calculate_mom_change(economic_data, 'm2_money_supply', ratio=False)
economic_data = calculate_mom_change(economic_data, 'industrial_production', ratio=False)

df=pd.read_csv(f"data/{ticker_lowercase}_news_sentiment.csv")

df['date'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S').dt.strftime('%Y-%m-%d')

daily_avg_sentiment_score = df.groupby('date')['ticker_sentiment_score'].mean().reset_index()
daily_avg_sentiment_score = daily_avg_sentiment_score.rename(columns={'ticker_sentiment_score': 'avg_sentiment_score'})
sentiment_counts = df.groupby(['date', 'ticker_sentiment_label']).size().unstack(fill_value=0)
sentiment_proportions = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
sentiment_proportions = sentiment_proportions.reset_index()
daily_sentiment_summary = pd.merge(daily_avg_sentiment_score, sentiment_proportions, on='date', how='left')

sentiment_mapping = {
    'Bearish': -1,
    'Somewhat-Bearish': -0.5,
    'Neutral': 0,
    'Somewhat-Bullish': 0.5,
    'Bullish': 1
}
daily_sentiment_summary['daily_sentiment_score'] = (
    daily_sentiment_summary['Bearish'] * sentiment_mapping['Bearish'] +
    daily_sentiment_summary['Somewhat-Bearish'] * sentiment_mapping['Somewhat-Bearish'] +
    daily_sentiment_summary['Neutral'] * sentiment_mapping['Neutral'] +
    daily_sentiment_summary['Somewhat-Bullish'] * sentiment_mapping['Somewhat-Bullish'] +
    daily_sentiment_summary['Bullish'] * sentiment_mapping['Bullish']
)
daily_sentiment_summary.rename(columns={'date': 'Date'}, inplace=True)
# Display the result

stock_data.columns = ['_'.join(filter(None, col)).strip() for col in stock_data.columns]
stock_data=stock_data.reset_index()
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
economic_data['Date'] = pd.to_datetime(economic_data['Date'])
daily_sentiment_summary['Date'] = pd.to_datetime(daily_sentiment_summary['Date'])
selected_columns = ['Date'] + [col for col in economic_data.columns if col.endswith('_mom')]

prepared_data = (
    stock_data.reset_index()
    .merge(economic_data[selected_columns], on='Date', how='left')  # Join with table2
    .merge(daily_sentiment_summary[['Date', 'daily_sentiment_score']], on='Date', how='left')  # Join with table3
)

prepared_data.to_csv(f'data/{ticker_lowercase}_prepared_data.csv', index=False)