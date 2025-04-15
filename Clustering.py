
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
    # Randomly select K initial centroids
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

    return clusters, centroids

def initialize_membership(n_clusters, n_samples):
    U = np.random.dirichlet(np.ones(n_clusters), size=n_samples).T
    return U

def calculate_cluster_centers(U, X, m):
    num = np.dot(U ** m, X)
    den = np.sum(U ** m, axis=1, keepdims=True)
    return num / den

def update_membership(X, centers, m):
    n_clusters = centers.shape[0]
    n_samples = X.shape[0]
    U_new = np.zeros((n_clusters, n_samples))

    for i in range(n_clusters):
        for j in range(n_samples):
            denom = sum(
                (np.linalg.norm(X[j] - centers[i]) / np.linalg.norm(X[j] - centers[k])) ** (2 / (m - 1))
                for k in range(n_clusters)
            )
            U_new[i, j] = 1 / denom
    return U_new

def fuzzy_c_means(X, n_clusters=3, m=2.0, max_iter=100, error=1e-5):
    X = np.array(X)
    n_samples = X.shape[0]
    U = initialize_membership(n_clusters, n_samples)

    for iteration in range(max_iter):
        centers = calculate_cluster_centers(U, X, m)
        U_new = update_membership(X, centers, m)

        if np.linalg.norm(U_new - U) < error:
            break
        U = U_new

    labels = np.argmax(U, axis=0)
    return centers, U, labels

def print_cluster_info(clusters, centroids, feature_names):
    print("\nCluster Centroids:")
    for i, centroid in enumerate(centroids):
        print(f"Cluster {i + 1}:", end=" ")
        for j, feature in enumerate(feature_names):
            print(f"{feature}={centroid[j]:.4f}", end="  ")
        print()

    print("\nCluster Distribution:")
    unique, counts = np.unique(clusters, return_counts=True)
    for i, count in zip(unique, counts):
        print(f"Cluster {i + 1}: {count} points ({count / len(clusters) * 100:.1f}%)")

data = read_file("Longotor1delta.xls")
normalized_data = z_score_norm(data)

# K-Means Clustering
# q is Minkowski distance: 1 = Manhattan, 2 = Euclidean
print("Running K-Means Clustering")
clusters, centroids = k_means_cluster(normalized_data, k=3, q=1, max_iterations=100)
feature_names = ['sch9/wt', 'ras2/wt', 'tor1/wt']
print_cluster_info(clusters, centroids, feature_names)
print()

# Fuzzy-Means Clustering
print("Running Fuzzy-Means Clustering")
centers, U, labels = fuzzy_c_means(normalized_data[feature_names].values, n_clusters=3)
#print(centers, U, labels)
labels = np.argmax(U, axis=0)
print_cluster_info(labels, centers, feature_names) # TODO: Replace w/ new implementation
