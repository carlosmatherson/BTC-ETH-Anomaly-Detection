import pandas as pd

def find_neighbors(df, point_idx, eps, features):
    
    # get point values
    point = df.loc[point_idx, features]

    # calculate distance (norm) on features
    distances = ((df[features] - point) ** 2).sum(axis=1).pow(0.5)

    return distances[distances <= eps].index

def dbscan(df, eps, min_pts, features):

    # start all labels as unassigned
    labels = pd.Series(index=df.index, data = -2) # mark unnasigned with -2
    cluster_id = 0 # first cluster id is 0

    # go through each point
    for idx in df.index:

        # skip assigned points
        if labels[idx] != -2:
            continue

        # find neighbords
        neighbors = find_neighbors(df, idx, eps, features)

        # mark noise with -1
        if len(neighbors) < min_pts:
            labels[idx] = -1
            continue

        # start next cluster
        cluster_id += 1
        labels[idx] = cluster_id

        # check each neighbor
        neighbors = neighbors.to_list()
        i = 0

        # len neighbors may change, check all recurseively 
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            # add noisey neighbor to cluster (border point)
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            # add unassigned nieghbors to cluster, then check their neighbors
            if labels[neighbor_idx] == -2:
                labels[neighbor_idx] = cluster_id
                new_neighbors = find_neighbors(df, neighbor_idx, eps, features)

                # add qualifying neighbors to current neighbor list 
                if len(new_neighbors) >= min_pts:
                    new_neighbors = new_neighbors.to_list()
                    neighbors.extend(n for n in new_neighbors if n not in neighbors)
            
            i += 1

    return labels

def main():

    df = pd.read_csv('../2_preprocess/processed_data.csv', parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    features = ['eth_beta_120D_norm', 'correlation_60D_norm', 'volatility_ratio_7D_norm', 'return_spread']
    
    # random params for initial testing
    labels = dbscan(df, eps=0.415, min_pts=2*len(features), features=features)
    
    # add labels to df
    df['cluster'] = labels
    
    # print stats
    n_clusters = len(labels[labels >= 0].unique())
    print(f"found {n_clusters} clusters")

    n_noise = (labels == -1).sum()
    print(f"found {n_noise} noises")
    
    df.to_csv('dbscan_results.csv')
    
    return df, labels

if __name__ == '__main__':
    main()