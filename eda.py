import pandas as pd
import matplotlib.pyplot as plt

def load_data(filepath='processed_data.csv'):

    # load data
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    # separate data into chosen features (final features for DBSCAN)
    chosen_features = ['eth_beta_120D_norm', 'correlation_60D_norm', 'volatility_ratio_7D_norm', 'return_spread']

    # separate out all normalized features (for feature analysis)
    normalized_features = [col for col in df.columns if col.endswith('_norm')]

    return df, normalized_features, chosen_features

def plot_distributions(df, features, title):

    # plot normalized feature distributions
    n_raw = len(features)
    fig, axes = plt.subplots(n_raw//2 + n_raw%2, 2, figsize=(15, 4*n_raw//2))
    fig.suptitle(title)
    
    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2
        df[feature].hist(ax=axes[row, col], bins=50)
        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{title}.png')
    plt.show()

def plot_correlation_matrix(df, features, title):

    # calcualte fearture correlation
    corr = df[features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    # add correlation values
    for i in range(len(features)):
        for j in range(len(features)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                         ha="center", va="center", color="black")
    
    # set ticks and labels
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels(features,)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'visualizations/{title}.png')
    plt.show()

def plot_scatter_matrix(df, features, title):

    # plot scatter matrix to look for natural clusters
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(20, 20))
    # big grid for all features
    #fig, axes = plt.subplots(n, n, figsize=(50, 50))
    fig.suptitle(title)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                axes[i, j].scatter(df[features[j]], df[features[i]], alpha=0.5, s=1)
            else:
                axes[i, j].hist(df[features[i]], bins=50)
            
            if i == n-1:
                axes[i, j].set_xlabel(features[j])
            if j == 0:
                axes[i, j].set_ylabel(features[i])
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{title}.png')
    plt.show()

def compute_kdistance_plot(df, features, k, title):

    # compute pairwise distances to find good value for eps
    distances = pd.DataFrame(index=df.index, columns=df.index)
    for i in df.index:
        diff = df.loc[i, features] - df[features]
        distances.loc[i] = (diff ** 2).sum(axis=1).pow(0.5)
    
    # get k-distances for each point and sort them
    kdistances = distances.apply(lambda x: x.sort_values().iloc[k])
    kdistances_sorted = kdistances.sort_values(ascending=True)
    
    # plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(kdistances_sorted)), kdistances_sorted.values)
    
    # add mean and percentile lines
    plt.axhline(y=kdistances_sorted.mean(), color='r', linestyle='--', 
                label=f'Mean {k}-distance')
    
    for p, c in zip([25, 50, 75], ['g', 'y', 'orange']):
        eps = kdistances_sorted.quantile(p/100)
        plt.axhline(y=eps, color=c, linestyle=':', 
                   label=f'{p}th percentile')
    
    plt.title(f'{k}-Distance Plot')
    plt.xlabel('Points (sorted by distance)')
    plt.ylabel(f'{k}-Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualizations/{k}{title}.png')
    plt.show()

    
    # Print potential eps values
    print("\nPotential eps values at different percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"{p}th percentile: {kdistances_sorted.quantile(p/100):.3f}")

def main():
    # Load data
    df, normalized_features, chosen_features = load_data()

    # Feature search
    # plot_distributions(df, normalized_features, title='All Normalized Features Distributions')
    # plot_correlation_matrix(df, normalized_features, title='All Normalized Feature Correlations')
    # plot_scatter_matrix(df, normalized_features, title='All Normalized Features Scatter Matrix')

    
    # Plot distributions
    plot_distributions(df, chosen_features, title='Norm Features Distributions')
    
    # Plot correlation matrices
    plot_correlation_matrix(df, chosen_features, title='Normalized Feature Correlations')
    
    # Plot scatter matrix for normalized features
    plot_scatter_matrix(df, chosen_features, title='Normalized Features Scatter Matrix')

    # Generate k-distance plot
    compute_kdistance_plot(df, normalized_features, k=2*len(chosen_features), title='distanceplot_final')
    
    # Print summary statistics
    print("\nAll Normalized Feature Summary Statistics:")
    print(df[normalized_features].describe())


if __name__ == "__main__":
    main()