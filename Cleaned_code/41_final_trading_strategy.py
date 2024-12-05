# pick the model with highest return

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)


def process_ticker_data(ticker, variable_methods, model_names):
     """
     Processes the ticker data and calculates the thresholds and dataframe for analysis.

     Parameters:
         ticker (str): The ticker symbol.
         variable_methods (list): List of variable methods (e.g., ['price_alone']).
         model_names (list): List of model names (e.g., ['lstm']).

     Returns:
         threshold_long (float): The threshold for long positions.
         threshold_short (float): The threshold for short positions.
         df (DataFrame): Processed DataFrame with predictions and actual values.
     """
     # Load the combined predictions file
     predictions_file = f'results/{ticker}_combined_predictions.csv'
     data_file = f'data/{ticker}_prepared_data.csv'
     quantiles_file = f'results/{ticker}_{variable_methods}_{model_names}_predicted_quantiles.csv'

     # Load predictions and data
     predictions = pd.read_csv(predictions_file)
     predictions = predictions[['Date', 'Actual', f'{variable_methods}_{model_names}_Predicted']]

     data = pd.read_csv(data_file)
     data = data.fillna(0).drop(columns=['level_0', 'index'], errors='ignore')

     # Load quantiles and calculate thresholds
     threshold_file = pd.read_csv(quantiles_file)
     threshold_long = threshold_file[threshold_file['Quantile'] == 0.75]['Value'].values[0]
     threshold_short = threshold_file[threshold_file['Quantile'] == 0.1]['Value'].values[0]

     # Merge and process data
     df = pd.merge(predictions[['Date', f'{variable_methods}_{model_names}_Predicted']],
                   data[['Date', 'Next_day_Adj_Open_price']], on='Date', how='left')
     df = df.rename(columns={
         'Next_day_Adj_Open_price': f'{ticker}_Actual',
         f'{variable_methods}_{model_names}_Predicted': f'{ticker}_Predicted'
     }).dropna()

     # Calculate percentage change in predicted and actual prices
     df[f'{ticker}_predict_pct_change'] = df[f'{ticker}_Predicted'].pct_change().shift(-1).fillna(0) * 100

     return threshold_long, threshold_short, df


voo_threshold_long, voo_threshold_short, voo_df=process_ticker_data('voo', 'price_alone', 'lstm_xgboost')
tsla_threshold_long, tsla_threshold_short, tsla_df=process_ticker_data('tsla', 'all_variables', 'arimax')
avgo_threshold_long, avgo_threshold_short, avgo_df=process_ticker_data('avgo', 'price_alone', 'lstm')
cof_threshold_long, cof_threshold_short, cof_df=process_ticker_data('cof', 'price_alone', 'lstm_xgboost')
crm_threshold_long, crm_threshold_short, crm_df=process_ticker_data('crm', 'all_variables', 'lstm')
gm_threshold_long, gm_threshold_short, gm_df=process_ticker_data('gm', 'price_alone', 'lstm')
gs_threshold_long, gs_threshold_short, gs_df=process_ticker_data('gs', 'all_variables', 'arimax')
ibm_threshold_long, ibm_threshold_short, ibm_df=process_ticker_data('ibm', 'all_variables', 'arimax')
ilmn_threshold_long, ilmn_threshold_short, ilmn_df=process_ticker_data('ilmn', 'all_variables', 'lstm')
nke_threshold_long, nke_threshold_short, nke_df=process_ticker_data('nke', 'price_alone', 'lstm')
nvda_threshold_long, nvda_threshold_short, nvda_df=process_ticker_data('nvda', 'all_variables', 'arimax')
orcl_threshold_long, orcl_threshold_short, orcl_df=process_ticker_data('orcl', 'all_variables', 'arimax')
regn_threshold_long, regn_threshold_short, regn_df=process_ticker_data('regn', 'all_variables', 'arimax')

all_predicted_df=voo_df.merge(tsla_df, on='Date', how='left')\
.merge(avgo_df, on='Date', how='left')\
.merge(cof_df, on='Date', how='left')\
.merge(crm_df, on='Date', how='left')\
.merge(gm_df, on='Date', how='left')\
.merge(gs_df, on='Date', how='left')\
.merge(ibm_df, on='Date', how='left')\
.merge(ilmn_df, on='Date', how='left')\
.merge(nke_df, on='Date', how='left')\
.merge(nvda_df, on='Date', how='left')\
.merge(orcl_df, on='Date', how='left')\
.merge(regn_df, on='Date', how='left')

all_predicted_df.to_csv("results/all_predicted_df.csv", index=False)

# Base Strategy:
# Parameters
capital = 100  # Initial investment
df = all_predicted_df

# Base Strategy: Buy $100 VOO and hold until the last day
capital_base = capital  # Start with $100
position_base = capital / df.loc[0, 'voo_Actual']  # Number of shares bought on the first day
capital_history_base = [capital_base]  # Track capital over time

# Calculate capital over time for the base strategy
for i in range(1, len(df)):
    capital_base = position_base * df.loc[i, 'voo_Actual']
    capital_history_base.append(capital_base)

print(capital_base)
# Save capital history for the base strategy
pd.Series(capital_history_base).to_csv("results/trading_strategy_base.csv", index=False)
print("Base strategy capital history saved to 'results/trading_strategy_base.csv'")

import numpy as np
def calculate_performance_metrics(capital_history):
     """
     Calculate performance metrics for a given capital history.

     Parameters:
         capital_history (list or np.array): Historical capital values.

     Returns:
         pd.DataFrame: A DataFrame containing total return, VaR (95%), max drawdown, and annualized volatility.
     """
     capital_history = np.array(capital_history)

     # Total Return
     initial_capital = capital_history[0]
     final_capital = capital_history[-1]
     total_return = (final_capital - initial_capital) / initial_capital

     # Daily Returns
     daily_returns = np.diff(capital_history) / capital_history[:-1]

     # VaR at 95%
     VaR_95 = np.percentile(daily_returns, 5)

     # Maximum Drawdown
     cumulative_max = np.maximum.accumulate(capital_history)
     drawdown = (cumulative_max - capital_history) / cumulative_max
     max_drawdown = np.max(drawdown)

     # Annualized Volatility
     volatility = np.std(daily_returns) * np.sqrt(252)

     # Create DataFrame
     performance_metrics = pd.DataFrame({
         'Total Return': [total_return],
         'VaR (95%)': [VaR_95],
         'Max Drawdown': [max_drawdown],
         'Volatility (Annualized)': [volatility]
     })

     return performance_metrics


performance_metrics_base = calculate_performance_metrics(capital_history_base)
performance_metrics_base.to_csv("results/trading_strategy_performance_metrics_base.csv", index=False)
print("Base strategy capital history saved to 'results/trading_strategy_performance_metrics_base.csv'")



#Alternative Strategy - low/medium/high risk

# Parameters

thresholds_long = {
    'voo': voo_threshold_long,
    'tsla': tsla_threshold_long,
    'avgo': avgo_threshold_long,
    'cof': cof_threshold_long,
    'crm': crm_threshold_long,
    'gm': gm_threshold_long,
    'gs': gs_threshold_long,
    'ibm': ibm_threshold_long,
    'ilmn': ilmn_threshold_long,
    'nke': nke_threshold_long,
    'nvda': nvda_threshold_long,
    'orcl': orcl_threshold_long,
    'regn': regn_threshold_long

}  # Long thresholds for each stock
thresholds_short = {
    'voo': voo_threshold_short
}  # Short thresholds for VOO

capital = 100  # Start with $100 in VOO
df = all_predicted_df  # Replace with your DataFrame containing the relevant columns

def trading_strategy(trading_invest_pct,thresholds_long,voo_threshold_short,hot_stock,capital,min_voo,df,risk_level):
    # Initialize positions for each stock
    positions = {stock: 0 for stock in ['voo','tsla','avgo','cof','crm','gm','gs','ibm','ilmn','nke','nvda','orcl','regn']}  # Initial positions for all stocks
    positions['voo'] = capital / df.loc[0, 'voo_Actual']  # Fully invested in VOO initially

    capital_history = []  # Track total capital over time
    investment_history = {stock: [] for stock in positions.keys()}  # Track daily investments for all stocks

    # Alternative Strategy
    for i in range(len(df)):
        actuals = {stock: df.loc[i, f'{stock}_Actual'] for stock in positions.keys()}
        pct_changes = {stock: df.loc[i, f'{stock}_predict_pct_change'] for stock in positions.keys()}

        # Current capital allocation
        values = {stock: positions[stock] * actuals[stock] for stock in positions.keys()}
        total_capital = sum(values.values())
        eligible_stocks = {
            stock: pct_changes[stock]
            for stock in thresholds_long
            if (stock in hot_stock and pct_changes[stock] > 0) or
               (stock not in hot_stock and pct_changes[stock] >= thresholds_long[stock])
        }
        # Enforce min_voo constraint
        if values['voo'] / total_capital >= min_voo:
            # Case 1: VOO pct_change < voo_threshold_short
            if pct_changes['voo'] < voo_threshold_short:
                if eligible_stocks:
                    total_pct_change = sum(eligible_stocks.values())
                    proportions = {stock: pct_changes[stock] / total_pct_change for stock in eligible_stocks}

                    sell_amount_voo = positions['voo'] * (trading_invest_pct / 100)
                    positions['voo'] -= sell_amount_voo

                    for stock, proportion in proportions.items():
                        buy_amount = sell_amount_voo * actuals['voo'] * proportion / actuals[stock]
                        positions[stock] += buy_amount

            # Case 2: VOO pct_change is between voo_threshold_short and 0
            elif voo_threshold_short <= pct_changes['voo'] < 0:
                if eligible_stocks:
                    total_pct_change = sum(eligible_stocks.values())
                    proportions = {stock: pct_changes[stock] / total_pct_change for stock in eligible_stocks}

                    sell_amount_voo = positions['voo'] * (trading_invest_pct * 0.5 / 100)
                    positions['voo'] -= sell_amount_voo

                    for stock, proportion in proportions.items():
                        buy_amount = sell_amount_voo * actuals['voo'] * proportion / actuals[stock]
                        positions[stock] += buy_amount

            # Case 3: VOO pct_change > 0
            elif pct_changes['voo'] >= 0:
                if eligible_stocks:
                    total_pct_change = sum(eligible_stocks.values())
                    proportions = {stock: pct_changes[stock] / total_pct_change for stock in eligible_stocks}

                    sell_amount_voo = positions['voo'] * (trading_invest_pct / 100)
                    positions['voo'] -= sell_amount_voo

                    for stock, proportion in proportions.items():
                        buy_amount = sell_amount_voo * actuals['voo'] * proportion / actuals[stock]
                        positions[stock] += buy_amount

        # Sell all positions of a stock if it is not in eligible_stocks

        for stock in thresholds_long.keys():
            if stock not in eligible_stocks and positions[stock] > 0:
                sell_amount = positions[stock]
                positions[stock] = 0
                buy_amount_voo = sell_amount * actuals[stock] / actuals['voo']
                positions['voo'] += buy_amount_voo

        # Update daily investments
        for stock in positions.keys():
            values[stock] = positions[stock] * actuals[stock]
            investment_history[stock].append(values[stock])

        total_capital = sum(values.values())
        capital_history.append(total_capital)


    # Output final capital
    final_capital = capital_history[-1]
    print(f"Final Capital: ${final_capital:.2f}")

    # Save results
    results_df = pd.DataFrame({
        'Date': df['Date'],
        'Total_Capital': capital_history,
        **{f'{stock}_Capital': investment_history[stock] for stock in positions.keys()}
    })
    results_df.to_csv(f"results/trading_strategy_{risk_level}_risk.csv", index=False)
    # Calculate the average portion for each stock
    for stock in positions.keys():
        results_df[f'{stock}_Portion'] = results_df[f'{stock}_Capital'] / results_df['Total_Capital']

    average_portions = results_df[[f'{stock}_Portion' for stock in positions.keys()]].mean().rename('Average Portion')
    average_portions_df = average_portions.reset_index()
    average_portions_df.columns = ['Stock', 'Average Portion']

    # Save to CSV
    average_portions_df.to_csv(f"results/trading_strategy_average_portions_{risk_level}_risk.csv", index=False)

    performance_metrics = calculate_performance_metrics(capital_history)
    performance_metrics.to_csv(f"results/trading_strategy_performance_metrics_{risk_level}_risk.csv", index=False)

trading_strategy(10,thresholds_long,voo_threshold_short,('') ,capital,0.2,df,'low')

trading_strategy(20,thresholds_long,voo_threshold_short,('tsla','nvda')  ,capital,0.2,df,'medium')

trading_strategy(50,thresholds_long,voo_threshold_short,('tsla','nvda')  ,capital,0.5,df,'high')
