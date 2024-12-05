


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('results/orcl_combined_predictions.csv')


# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Find the last available date for each month
last_available_dates = data.groupby(data['Date'].dt.to_period('M'))['Date'].max()

# Plot the data
plt.figure(figsize=(12, 6))

# Plot 'Actual' with a distinct line style and color
plt.plot(data['Date'], data['Actual'], label='Actual', linewidth=2.5, color='black')

# Plot the prediction columns with different colors
prediction_columns = [col for col in data.columns if 'Predicted' in col]
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta']

for i, col in enumerate(prediction_columns):
    plt.plot(data['Date'], data[col], label=col, linewidth=1.5, color=colors[i % len(colors)])

# Formatting the plot
plt.title('Actual vs Predicted Prices for ORCL', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)

# Show only the last available date for each month
plt.xticks(last_available_dates, labels=last_available_dates.dt.strftime('%Y-%m-%d'), rotation=45)

plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)

# Display the plot
plt.tight_layout()
plt.savefig('figures/orcl_actual_vs_predicted.png', dpi=300)
plt.show(block=True)

import pandas as pd
import matplotlib.pyplot as plt



def plot_predictions(tickers):


    for ticker in tickers:
        # Load the data for the current ticker
        data = pd.read_csv(f'results/{ticker}_combined_predictions.csv')

        # Convert 'Date' to datetime
        data['Date'] = pd.to_datetime(data['Date'])

        # Find the last available date for each month
        last_available_dates = data.groupby(data['Date'].dt.to_period('M'))['Date'].max()

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot 'Actual' with a distinct line style and color
        plt.plot(data['Date'], data['Actual'], label='Actual', linewidth=2.5, color='black')

        # Plot the prediction columns with different colors
        prediction_columns = [col for col in data.columns if 'Predicted' in col]
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta']

        for i, col in enumerate(prediction_columns):
            plt.plot(data['Date'], data[col], label=col, linewidth=1.5, color=colors[i % len(colors)])

        # Formatting the plot
        plt.title(f'Actual vs Predicted Prices for {ticker.upper()}', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)

        # Show only the last available date for each month
        plt.xticks(last_available_dates, labels=last_available_dates.dt.strftime('%Y-%m-%d'), rotation=45)

        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'figures/{ticker}_actual_vs_predicted.png', dpi=300)
        plt.close()  # Close the plot to save memory and avoid overlap during the loop


# Define the tickers
tickers = ['voo', 'tsla', 'avgo', 'cof', 'crm', 'gm', 'gs', 'ibm', 'ilmn', 'nke', 'nvda', 'orcl', 'regn']

# Call the function
plot_predictions(tickers)