
import sys
import numpy as np
import pandas as pd

def read_file(filename):
    try:
        return pd.read_excel(filename, usecols="A,D,E,F")
    except PermissionError:
        print("[Error:] Cannot read from the file while it is open. Please close it.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"[Error:] Cannot find the file '{filename}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Error:] An unknown error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def z_score_norm(pandas_df):
    normalized_df = pandas_df.copy()

    # Z-score norm for e/a column
    for column in ["sch9/wt", "ras2/wt", "tor1/wt"]:
        normalized_df[column] = (normalized_df[column] - normalized_df[column].mean()) / normalized_df[column].std()

    return normalized_df

def calc_minkowski_dist(i, j, q):
    abs_diff_power_sum = 0

    for idx in range(len(i)):
        abs_diff_power_sum += abs(i[idx] - j[idx]) ** q

    return abs_diff_power_sum ** (1 / q)

def k_means_cluster(pandas_df, k, q, max_iterations):
    # Convert pandas df columns into numpy array
    features = pandas_df[['sch9/wt', 'ras2/wt', 'tor1/wt']].values
    num_features = len(features)

    # 1. Partition into K subsets
    np.random.seed(42)
    # Randomly split data into K initial centroids
    random_indices = np.random.choice(num_features, k, replace=False)
    centroids = features[random_indices]

    clusters = np.zeros(num_features, dtype=int)
    prev_clusters = np.ones(num_features, dtype=int)

    iteration = 0

    # Run until convergence or max iterations reached
    while not np.array_equal(clusters, prev_clusters) and iteration < max_iterations:
        prev_clusters = np.copy(clusters)

        # 3. Assign e/a datapoint to nearest centroid
        for idx in range(num_features):
            # Calc Minkowski dist to e/a centroid
            distances = [calc_minkowski_dist(features[idx], centroid, q) for centroid in centroids]
            # Assign datapoint to closest centroid (argmin returns idx w/ smallest distance)
            clusters[idx] = np.argmin(distances)

        # 2. Calc current cluster's centroids
        for idx2 in range(k):
            # Datapoints belonging to this cluster
            cluster_points = features[clusters == idx2]
            # Update centroid's mean
            if len(cluster_points > 0):
                centroids[idx2] = np.mean(cluster_points, axis=0)

        iteration += 1

    if iteration <= max_iterations:
        print(f"K-means converged after {iteration} iterations.")
    else:
        print(f"K-means reached max iteration count ({max_iterations}). Stopping w/ current vals.")

    # Tmp
    one = features[clusters == 0]
    two = features[clusters == 1]
    three = features[clusters == 2]
    return clusters, centroids

data = read_file("Longotor1delta.xls")
normalized_data = z_score_norm(data)
# q is Minkowski distance: 1 = Manhattan, 2 = Euclidean
clusters, centroids = k_means_cluster(normalized_data, k=3, q=1, max_iterations=100)

print("Clusters and first 10 centroids:")
for i in range(len(clusters)):
    print(f"Cluster: {i + 1}")
    print(f"Centroids: {centroids[:11]}")
