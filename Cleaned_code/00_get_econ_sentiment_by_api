
# economic_Data
# https://fred.stlouisfed.org/docs/api/api_key.html


import yfinance as yf
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)

start_date = "2022-01-01"
end_date = "2024-11-22"

import requests
import pandas as pd

# Your FRED API Key, replace fred_demo to api key
API_KEY = 'fred_demo'




TICKER_INDICATORS = {
    'tsla': {
        "interest_rate": "DFF",  # Effective Federal Funds Rate
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate
        "cpi": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP (percent change from preceding period)
        "industrial_production": "INDPRO",  # Industrial Production Index
        "vehicle_sales": "TOTALSA",  # Total Vehicle Sales
        "personal_consumption": "PCE",  # Personal Consumption Expenditures
        "retail_sales_auto": "MRTSSM441USN",  # Retail Sales: Motor Vehicle and Parts Dealers
        "crude_oil_prices": "DCOILWTICO",  # Crude Oil Prices
        "sp500": "SP500"  # S&P 500 Index
    },
    'voo': {
        "interest_rate": "DFF",  # Effective Federal Funds Rate
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate
        "cpi": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP (percent change from preceding period)
        "industrial_production": "INDPRO",  # Industrial Production Index
        "personal_consumption": "PCE",  # Personal Consumption Expenditures
        "housing_starts": "HOUST",  # Housing Starts
        "retail_sales": "RSAFS",  # Retail Sales: Total
        "sp500": "SP500",  # S&P 500 Index
        "vix": "VIXCLS"  # CBOE Volatility Index (Market Volatility)
    },
    'nvda': {
        "interest_rate": "DFF",  # Effective Federal Funds Rate (borrowing cost)
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate (consumer demand)
        "cpi": "CPIAUCSL",  # Consumer Price Index (inflation)
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth (economic activity)
        "industrial_production": "INDPRO",  # Industrial Production Index
        "personal_consumption": "PCE",  # Personal Consumption Expenditures (consumer demand)
        "technology_capital_spending": "ANDENO",
        # # Orders of Nondefense Capital Goods Excluding Aircraft (tech investment proxy)
        "crude_oil_prices": "DCOILWTICO",  # Crude Oil Prices (supply chain costs)
        "semiconductor_shipments": "IPG3361T3S",  # Semiconductor and Electronics Shipment Index
    },
    'cof': {
        "interest_rate": "DFF",  # Effective Federal Funds Rate (borrowing/lending cost)
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate (consumer demand/credit risk)
        "cpi": "CPIAUCSL",  # Consumer Price Index (inflation affecting spending)
        "revolving_credit": "REVOLSL",  # Total Revolving Credit (credit card activity proxy)
        "delinquency_rate": "DRCCLACBS",  # Delinquency Rate on Credit Card Loans
        "personal_savings_rate": "PSAVERT",  # Personal Savings Rate
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth (economic activity)
        "consumer_sentiment": "UMCSENT",  # University of Michigan Consumer Sentiment
        "housing_starts": "HOUST",  # Housing Starts (proxy for consumer confidence)
        "retail_sales": "RSAFS",  # Total Retail Sales
    },
    'gs': {
        "interest_rate": "DFF",  # Effective Federal Funds Rate (borrowing/lending costs)
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate (economic health)
        "cpi": "CPIAUCSL",  # Consumer Price Index (inflation)
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth (economic activity)
        "corporate_bond_spread": "BAA10Y",  # Moody's BAA Corporate Bond Yield Relative to 10-Year Treasury
        "stock_market_volatility": "VIXCLS",  # CBOE Volatility Index (market risk/fear)
        "industrial_production": "INDPRO",  # Industrial Production Index
        "m2_money_supply": "M2SL",  # M2 Money Stock (liquidity in the economy)
        "commercial_loans": "BUSLOANS",  # Commercial and Industrial Loans Outstanding
        "sp500": "SP500",  # S&P 500 Index (overall market performance)
    },
    'ibm': {
        "interest_rate": "DFF",  # Effective Federal Funds Rate
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth
        "industrial_production": "INDPRO",  # Industrial Production Index
        "business_investment": "ANDENO",  # Orders of Nondefense Capital Goods Excluding Aircraft
        "enterprise_software_spending": "PCEDG",
        # Personal Consumption Expenditures: Durable Goods (proxy for IT hardware/software)
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate
        "global_trade_volume": "EXPCH",  # US Exports of Goods (proxy for global demand)
        "corporate_earnings": "CPATAX",  # Corporate Profits After Tax
    },
    'avgo': {
        "semiconductor_shipments": "IPG3361T3S",  # Semiconductor and other electronic component shipments
        # "tech_sector_index": "NDXT",  # Nasdaq-100 Technology Sector Index
        "industrial_production": "INDPRO",  # Industrial Production Index
        "global_trade_volume": "EXPCH",  # US Exports of Goods (proxy for global demand)
        "crude_oil_prices": "DCOILWTICO",  # Crude Oil Prices (proxy for transportation and logistics costs)
        "interest_rate": "DFF",  # Effective Federal Funds Rate
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth
        "m2_money_supply": "M2SL",  # M2 Money Stock (liquidity in the economy)
        "personal_computing_sales": "PCU334111334111P",  # Personal Computer Manufacturing (proxy for chip demand)
        "consumer_sentiment": "UMCSENT",  # University of Michigan Consumer Sentiment Index
    },
    'crm': {
        "business_investment": "ANDENO",  # Orders of Nondefense Capital Goods Excluding Aircraft
        # "tech_sector_index": "NDXT",  # Nasdaq-100 Technology Sector Index
        "software_spending": "PCES",
        # Personal Consumption Expenditures: Services (proxy for enterprise software demand)
        "interest_rate": "DFF",  # Effective Federal Funds Rate
        "corporate_earnings": "CPATAX",  # Corporate Profits After Tax
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate
        "consumer_sentiment": "UMCSENT",  # University of Michigan Consumer Sentiment
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth
        "m2_money_supply": "M2SL",  # M2 Money Stock (economic liquidity)
        "industrial_production": "INDPRO",  # Industrial Production Index (broad business activity proxy)
    },
    'orcl': {
        "business_investment": "ANDENO",  # Orders of Nondefense Capital Goods Excluding Aircraft
        # "tech_sector_index": "NDXT",  # Nasdaq-100 Technology Sector Index
        "software_spending": "PCES",  # Personal Consumption Expenditures: Services
        "corporate_earnings": "CPATAX",  # Corporate Profits After Tax
        "interest_rate": "DFF",  # Effective Federal Funds Rate
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth
        "industrial_production": "INDPRO",  # Industrial Production Index
        "m2_money_supply": "M2SL",  # M2 Money Stock (economic liquidity)
        # "cloud_services_spending": "AWSSOFT",  # Cloud Services Spending Proxy
        "consumer_sentiment": "UMCSENT",  # University of Michigan Consumer Sentiment Index
    },
    'nke': {
        "consumer_spending": "PCEC",  # Personal Consumption Expenditures
        "retail_sales": "RSAFS",  # Retail Sales: Total
        "import_prices": "IR",  # Import Price Index
        # "export_prices": "EP",  # Export Price Index
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate
        "consumer_sentiment": "UMCSENT",  # University of Michigan Consumer Sentiment Index
        "crude_oil_prices": "DCOILWTICO",  # Crude Oil Prices (proxy for logistics and transportation costs)
        "interest_rate": "DFF",  # Effective Federal Funds Rate
        "global_trade_volume": "EXPCH",  # US Exports of Goods
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth
    },
    'gm': {
        "vehicle_sales": "TOTALSA",  # Total Vehicle Sales
        "industrial_production": "INDPRO",  # Industrial Production Index
        "consumer_spending": "PCEC",  # Personal Consumption Expenditures
        "retail_sales_vehicles": "MRTSSM441USN",  # Retail Sales: Motor Vehicle and Parts Dealers
        "crude_oil_prices": "DCOILWTICO",  # Crude Oil Prices
        "interest_rate": "DFF",  # Effective Federal Funds Rate
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth
        # "steel_prices": "PCU3311103311101",  # Steel Mill Products (proxy for raw material costs)
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate
        "consumer_sentiment": "UMCSENT",  # University of Michigan Consumer Sentiment Index
    },
    'regn': {
        # "healthcare_spending": "HLTHS",  # Personal Consumption Expenditures: Healthcare
        # "research_and_development": "IRIP",  # Investment in Intellectual Property (proxy for R&D spending)
        "pharmaceutical_prices": "CUUR0000SETB01",  # Consumer Price Index: Prescription Drugs
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate
        "interest_rate": "DFF",  # Effective Federal Funds Rate
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth
        "consumer_sentiment": "UMCSENT",  # University of Michigan Consumer Sentiment Index
        # "biotech_index": "IBB",  # Nasdaq Biotechnology Index
        "m2_money_supply": "M2SL",  # M2 Money Stock (economic liquidity)
        # "health_insurance_coverage": "CMEDHHSI",  # Health Insurance Coverage
    },
    'ilmn': {
        # "healthcare_spending": "HLTHS",  # Personal Consumption Expenditures: Healthcare
        # "research_and_development": "IRIP",  # Investment in Intellectual Property (proxy for R&D spending)
        "pharmaceutical_prices": "CUUR0000SETB01",  # Consumer Price Index: Prescription Drugs
        "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate
        "interest_rate": "DFF",  # Effective Federal Funds Rate
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth
        "consumer_sentiment": "UMCSENT",  # University of Michigan Consumer Sentiment Index
        # "biotech_index": "IBB",  # Nasdaq Biotechnology Index
        "m2_money_supply": "M2SL",  # M2 Money Stock (economic liquidity)
        # "health_insurance_coverage": "CMEDHHSI",  # Health Insurance Coverage
    },


    # Add more tickers and their indicators here...
}

# Function to fetch data from FRED
def fetch_fred_data(series_id, start_date, end_date):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Parse data into DataFrame
    observations = data.get("observations", [])
    dates = [obs["date"] for obs in observations]
    values = [float(obs["value"]) if obs["value"] != "." else None for obs in observations]

    return pd.DataFrame({"Date": dates, series_id: values}).set_index("Date")


# Fetch each indicator and merge into a single DataFrame
def get_economic_indicators(ticker, indicators, start_date, end_date):
    data_frames = []
    for name, series_id in indicators.items():
        df = fetch_fred_data(series_id, start_date, end_date)
        df.columns = [name]  # Rename column to indicator name
        data_frames.append(df)

    # Merge all indicators on Date
    economic_data = pd.concat(data_frames, axis=1)
    economic_data.sort_index(inplace=True)
    economic_data = economic_data.ffill().bfill()  # Fill missing values
    return economic_data


# Save data for each ticker


for ticker, indicators in TICKER_INDICATORS.items():
    print(f"Processing {ticker}...")
    economic_data = get_economic_indicators(ticker, indicators, start_date, end_date)
    economic_data.reset_index(inplace=True)
    economic_data.to_csv(f"data/{ticker}_economic_data.csv", index=False)
    print(f"Saved data for {ticker} to data/{ticker}_economic_data.csv")



# sentiment_data


import requests
import pandas as pd
import time  # Optional, to avoid rate limits

ticker_list=['VOO','TSLA','AVGO','COF','CRM','GM','GS','IBM','ILMN','NKE','NVDA','ORCL','REGN']
# Initialize variables
# Your alphavantage API Key, replace alpha_demo to api key
apikey = 'alpha_demo'
base_url = 'https://www.alphavantage.co/query'
limit = 1000000
sort = 'EARLIEST'
# Set initial time_from value (e.g., start date)
time_from = '20220101T0000'
time_to = '20241123T0000'

for ticker in ticker_list:
    print(f"Processing {ticker}...")
    all_records = []


    for i in range(3):
        # Construct the URL with updated time_from
        url = f'{base_url}?function=NEWS_SENTIMENT&tickers={ticker}&time_from={time_from}&time_to={time_to}&sort={sort}&limit={limit}&apikey={apikey}'

        # Request data
        r = requests.get(url)
        data = r.json()

        # Check if 'feed' is present in the response
        if 'feed' not in data:
            print(f"No 'feed' data found in response for request {i + 1}")
            break

        # Extract relevant information
        records = []
        for item in data["feed"]:
            time_published = item.get("time_published")

            for sentiment in item.get("ticker_sentiment", []):
                if sentiment.get("ticker") == ticker:
                    ticker_sentiment_score = sentiment.get("ticker_sentiment_score")
                    ticker_sentiment_label = sentiment.get("ticker_sentiment_label")

                    # Append to records as a tuple
                    records.append((time_published, ticker_sentiment_score, ticker_sentiment_label))

        # Append the current records to all_records
        all_records.extend(records)

        # Update time_from with the last `time_published` in this response for the next iteration
        if records:
            time_from = records[-1][0][:-2]  # The last `time_published` value

        # Optional: Sleep to avoid hitting the API rate limit
        time.sleep(1)

    # Create DataFrame
    df = pd.DataFrame(all_records, columns=["time_published", "ticker_sentiment_score", "ticker_sentiment_label"])

    df.to_csv(f"data/{ticker}_news_sentiment.csv", index=False)
