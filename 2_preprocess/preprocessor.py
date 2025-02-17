import pandas as pd

def preprocess_data(btc_data='../1_data/btc_data.csv', eth_data='../1_data/eth_data.csv'):

    # read in data, parsing timestamps to datetime
    btc_df = pd.read_csv(btc_data, parse_dates=['timestamp'])
    eth_df = pd.read_csv(eth_data, parse_dates=['timestamp'])

    # set dates to index
    btc_df.set_index('timestamp', inplace=True)
    eth_df.set_index('timestamp', inplace=True)
    
    # filter out dates not in common
    dates_in_common = btc_df.index.intersection(eth_df.index)
    btc_df = btc_df.loc[dates_in_common]
    eth_df = eth_df.loc[dates_in_common]

    # create new data frame for storing features
    df = pd.DataFrame(index=dates_in_common)
    df['btc_price'] = btc_df['close']
    df['eth_price'] = eth_df['close']

    # calcualte BTC/ETH pair
    df['price_pair'] = df['btc_price'] / df['eth_price']

    # calcualte daily returns using close price
    df['btc_returns'] = df['btc_price'].pct_change()
    df['eth_returns'] = df['eth_price'].pct_change()

    # calculate return spread (directional difference in returns)
    df['return_spread'] = df['btc_returns'] - df['eth_returns']

    features_2_norm = ['return_spread']
    for window in [7,14,30,60,90,120]:
        # calculate volatility ratio (btc std / eth std)
        df[f'volatility_ratio_{window}D'] = df['btc_returns'].rolling(window).std() / df['eth_returns'].rolling(window).std()

        # Ethereum Beta calculate beta ()
        df[f'eth_beta_{window}D'] = df['eth_returns'].rolling(window).cov(df['btc_returns']) / df['btc_returns'].rolling(window).var()

        # correlation
        df[f'correlation_{window}D'] = df['btc_returns'].rolling(window).corr(df['eth_returns'])

        features_2_norm.extend([f'volatility_ratio_{window}D', f'eth_beta_{window}D', f'correlation_{window}D'])

    # normalize features
    for col in features_2_norm:
        df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df.dropna(inplace=True)

    return df


def main():

    df = preprocess_data()
    df.to_csv('processed_data.csv')

if __name__ == '__main__':
    main()