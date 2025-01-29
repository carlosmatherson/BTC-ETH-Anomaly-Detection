import pandas as pd

def preprocess_data(btc_data='btc_data.csv', eth_data='eth_data.csv'):

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

    # calcualte daily returns using close price
    df['btc_pct_change'] = df['btc_price'].pct_change()
    df['eth_pct_change'] = df['eth_price'].pct_change()
    
    # calcualte price ratio BTC/ETH
    df['price_ratio'] = df['btc_price']/df['eth_price']

    # calcaulte 30-day rolling return correlation
    df['pct_change_corr'] = df['btc_pct_change'].rolling(window=30).corr(df['eth_pct_change'])

    # calculate price ratio pct_change
    df['price_ratio_change'] = df['price_ratio'].pct_change()

    # collect feature names in a list
    features = ['btc_pct_change', 'eth_pct_change', 'price_ratio_change', 'pct_change_corr']

    # calcualte z-scores for features
    df = z_score_features(df, features)

    # normalize features
    df = normalize_features(df, features)

    df.dropna(inplace=True)

    return df

def normalize_features(df, feature_columns):

    for col in feature_columns:
        
        df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df
    
def z_score_features(df, feature_columns):

    for col in feature_columns:

        df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()

    return df

def main():

    df = preprocess_data()
    df.to_csv('processed_data.csv')

if __name__ == '__main__':
    main()