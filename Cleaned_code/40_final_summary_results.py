


import pandas as pd
import os

# Define the sets of variables
tickers = ['voo','tsla','avgo','cof','crm','gm','gs','ibm','ilmn','nke','nvda','orcl','regn']

variable_methods = ['all_variables', 'price_alone']
model_names = ['lstm', 'lstm_xgboost', 'lstm_attention', 'arimax']
strategies = ['long_only', 'both']

#combined metrics
# Loop through all combinations of ticker, variable_method, and model_name
for ticker in tickers:
    # Initialize an empty list to store DataFrames
    all_data = []
    for variable_method in variable_methods:
        for model_name in model_names:
            # Read the general metrics CSV
            try:
                metrics_file = f'results/{ticker}_{variable_method}_{model_name}_metrics_1.csv'
                df_base = pd.read_csv(metrics_file)
                if 'R²' in df_base.columns:
                    df_base = df_base.rename(columns={'R²': 'R2'})
            except FileNotFoundError:
                print(f"File not found: {metrics_file}")
                continue

            # Prefix columns for different strategies and merge
            strategy_dfs = []
            for strategy in strategies:
                try:
                    strategy_file = f'results/{ticker}_{variable_method}_{model_name}_{strategy}_metrics_2.csv'
                    df_strategy = pd.read_csv(strategy_file)

                    if 'Final Capital ($)' in df_strategy.columns:
                        df_strategy['Return (%)'] = ((df_strategy['Final Capital ($)'] - 100) / 100).apply(lambda x: f"{x:.1%}")
                    # Prefix columns with strategy name
                    df_strategy = df_strategy.rename(
                        columns=lambda x: f"{strategy}_{x}" if x not in ['ticker', 'variable_method', 'model_name'] else x
                    )

                    if f"{strategy}_trading_strategy" in df_strategy.columns:
                        df_strategy = df_strategy.drop(columns=[f"{strategy}_trading_strategy"])

                    strategy_dfs.append(df_strategy)
                except FileNotFoundError:
                    print(f"File not found: {strategy_file}")

            # Merge all strategy-specific DataFrames with the base DataFrame
            for strategy_df in strategy_dfs:
                df_base = df_base.merge(strategy_df, on=['ticker', 'variable_method', 'model_name'], how='outer')

            # Append the processed DataFrame to the list
            all_data.append(df_base)

    if all_data:
        ticker_df = pd.concat(all_data, ignore_index=True)
        # Reorder columns to ensure 'ticker', 'variable_method', 'model_name' are the first three columns
        cols = ['ticker', 'variable_method', 'model_name'] + [
            col for col in ticker_df.columns if col not in ['ticker', 'variable_method', 'model_name']
        ]
        ticker_df = ticker_df[cols]
        # Save the DataFrame for this ticker
        ticker_output_filename = f'results/{ticker}_combined_metrics.csv'
        ticker_df.to_csv(ticker_output_filename, index=False)
        print(f"Data for {ticker} saved to {ticker_output_filename}")

#combined prediction
# Loop through each ticker
for ticker in tickers:
    base_df = None  # Initialize the base DataFrame

    # Loop through variable methods and model names to create the combined file
    for variable_method in variable_methods:
        for model_name in model_names:
            try:
                # Construct the file name
                prediction_file = f'results/{ticker}_{variable_method}_{model_name}_predictions_vs_actuals.csv'

                # Read the CSV
                pred_df = pd.read_csv(prediction_file)

                # If base_df is not initialized, use the first file's Date and Actual columns
                if base_df is None:
                    base_df = pred_df[['Date', 'Actual']]

                # Rename the 'Predicted' column
                pred_df = pred_df.rename(columns={'Predicted': f'{variable_method}_{model_name}_Predicted'})

                # Add the prediction column to the base DataFrame
                base_df = base_df.merge(pred_df[['Date', f'{variable_method}_{model_name}_Predicted']], on='Date',
                                        how='left')
            except FileNotFoundError:
                print(f"File not found: {prediction_file}")
                continue

    # Save the combined DataFrame for this ticker
    output_filename = f'results/{ticker}_combined_predictions.csv'
    base_df.to_csv(output_filename, index=False)
    print(f"Combined predictions saved to {output_filename}")
