import pandas as pd
import matplotlib.pyplot as plt

def load_data(dbscan_results='../4_model/dbscan_results.csv'):
    # load data with dbscan results
    df = pd.read_csv(dbscan_results, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def statistical_anomalies(df, features, n_std=2.5):
    # find anomalies using statistical method (n standard deviations from mean)
    anomalies = pd.Series(False, index=df.index)
    
    for feature in features:
        mean = df[feature].mean()
        std = df[feature].std()
        threshold = n_std * std
        
        # mark as anomaly if outside n standard deviations
        feature_anomalies = (df[feature] < (mean - threshold)) | (df[feature] > (mean + threshold))
        anomalies = anomalies | feature_anomalies
        
    return anomalies

def plot_comprehensive_analysis(df, features):
    # get statistical anomalies 
    stat_anomalies = statistical_anomalies(df, features)
    
    # get dbscan anomalies (noise points)
    dbscan_anomalies = df['cluster'] == -1
    
    # agreement between methods
    agreement = (stat_anomalies & dbscan_anomalies)
    stat_only = (stat_anomalies & ~dbscan_anomalies)
    dbscan_only = (dbscan_anomalies & ~stat_anomalies)
    
    # create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))
    
    # plot BTC/ETH price ratio
    ax1.plot(df.index, df['price_pair'], label='BTC/ETH Ratio', color='gray', alpha=0.6)
    ax1.scatter(df[agreement].index, df[agreement]['price_pair'], 
                c='red', label='Both Methods', s=100, alpha=0.6)
    ax1.scatter(df[stat_only].index, df[stat_only]['price_pair'], 
                c='blue', label='Statistical Only', s=100, alpha=0.6)
    ax1.scatter(df[dbscan_only].index, df[dbscan_only]['price_pair'], 
                c='green', label='DBSCAN Only', s=100, alpha=0.6)
    ax1.set_title('BTC/ETH Price Ratio with Detected Anomalies')
    ax1.legend()
    ax1.grid(True)
    
    # plot BTC price
    ax2.plot(df.index, df['btc_price'], label='BTC Price', color='gray', alpha=0.6)
    ax2.scatter(df[agreement].index, df[agreement]['btc_price'], 
                c='red', label='Both Methods', s=100, alpha=0.6)
    ax2.scatter(df[stat_only].index, df[stat_only]['btc_price'], 
                c='blue', label='Statistical Only', s=100, alpha=0.6)
    ax2.scatter(df[dbscan_only].index, df[dbscan_only]['btc_price'], 
                c='green', label='DBSCAN Only', s=100, alpha=0.6)
    ax2.set_title('Bitcoin Price with Detected Anomalies')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: ETH price
    ax3.plot(df.index, df['eth_price'], label='ETH Price', color='gray', alpha=0.6)
    ax3.scatter(df[agreement].index, df[agreement]['eth_price'], 
                c='red', label='Both Methods', s=100, alpha=0.6)
    ax3.scatter(df[stat_only].index, df[stat_only]['eth_price'], 
                c='blue', label='Statistical Only', s=100, alpha=0.6)
    ax3.scatter(df[dbscan_only].index, df[dbscan_only]['eth_price'], 
                c='green', label='DBSCAN Only', s=100, alpha=0.6)
    ax3.set_title('Ethereum Price with Detected Anomalies')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'comprehensive_anomaly_analysis.png')
    plt.show()
    
    # print summary statistics
    print("\nAnomalies Detection Summary:")
    print(f"Total points analyzed: {len(df)}")
    print(f"Statistical anomalies: {stat_anomalies.sum()}")
    print(f"DBSCAN anomalies: {dbscan_anomalies.sum()}")
    print(f"Agreement between methods: {agreement.sum()}")
    print(f"Statistical only: {stat_only.sum()}")
    print(f"DBSCAN only: {dbscan_only.sum()}")
    
    # calc agreement rate
    agreement_rate = agreement.sum() / (stat_anomalies.sum() + dbscan_anomalies.sum() - agreement.sum())
    print(f"\nAgreement rate between methods: {agreement_rate:.2%}")
    
    # analyze anomaly distributions
    print("\nTemporal Analysis of Anomalies:")
    # group anomalies by month
    monthly_anomalies = pd.DataFrame({
        'Both': agreement.resample('M').sum(),
        'Statistical': stat_only.resample('M').sum(),
        'DBSCAN': dbscan_only.resample('M').sum()
    })
    print("\nMonths with highest number of anomalies:")
    print(monthly_anomalies.sum(axis=1).nlargest(5))

def main():
    # load data
    df = load_data()
    
    # features used for anomaly detection
    features = ['eth_beta_120D_norm', 'correlation_60D_norm', 
                'volatility_ratio_7D_norm', 'return_spread']
    
    # generate comprehensive analysis plots
    plot_comprehensive_analysis(df, features)

if __name__ == "__main__":
    main()