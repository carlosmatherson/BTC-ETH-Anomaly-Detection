import os
from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import pytz

def fetch_crypto_data(symbol_or_symbols, start_date, end_date):

    # Initialize Alpaca client for cryptocurrency data
    client = CryptoHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY")
    )
    
    # request params for alpaca crypto request
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbol_or_symbols,
        timeframe = TimeFrame.Day,
        start=start_date,
        end=end_date,
    )

    # get data and remove symbol from index
    bars = client.get_crypto_bars(request_params)
    df = bars.df
    df = df.reset_index(level=0, drop=True) # drop symbol column
    
    return df


def main():
    
    # configure date time format for start/end dates
    utc = pytz.UTC
    #end_date = datetime.now(utc)
    end_date = datetime(2025, 1, 1, tzinfo=utc)
    start_date = datetime(2021, 1, 1, tzinfo=utc)

    # fetch and save btc OHLCV data
    btc_df = fetch_crypto_data('BTC/USD', start_date, end_date)
    btc_df.to_csv('btc_data.csv', index=True)

    # fetch and save eth OHLCV data
    eth_df = fetch_crypto_data('ETH/USD', start_date, end_date)
    eth_df.to_csv('eth_data.csv', index=True)

if __name__ == "__main__":
    main()
    