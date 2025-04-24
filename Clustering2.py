import sys
import pandas as pd
import numpy as np

def read_file(filename):
    try:
        return pd.read_excel(filename, usecols="A,B,D,E,F")
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

    return clusters, centroids

#Randomly initialize centroids and the membership of the data points
def initialize_membership(n_clusters, n_samples):
    U = np.random.dirichlet(np.ones(n_clusters), size=n_samples).T
    return U


def calculate_cluster_centers(U, X, m):
    #Weighted data
    num = np.dot(U ** m, X)
    #weight of each cluster
    den = np.sum(U ** m, axis=1, keepdims=True)
    #Centroid
    return num / den

#Updates membership of data based on distance to the clusters
def update_membership(X, centers, m):
    n_clusters = centers.shape[0]
    n_samples = X.shape[0]
    U_new = np.zeros((n_clusters, n_samples))

    #Goes through each cluster and each data point
    for i in range(n_clusters):
        for j in range(n_samples):
            denom = sum(
                #calculates distance
                (np.linalg.norm(X[j] - centers[i]) / np.linalg.norm(X[j] - centers[k])) ** (2 / (m - 1))
                for k in range(n_clusters)
            )
            U_new[i, j] = 1 / denom
    return U_new

#Actual Fuzzy algorithm
def fuzzy_c_means(X, n_clusters, m=2.0, max_iter=100, error=1e-5):
    #takes in data and initializes the centroids and membership of the data points
    X = np.array(X)
    n_samples = X.shape[0]
    U = initialize_membership(n_clusters, n_samples)

    #calculate centroids and calculate membership
    for iteration in range(max_iter):
        centers = calculate_cluster_centers(U, X, m)
        U_new = update_membership(X, centers, m)

        #Determines if convergence is reached
        if np.linalg.norm(U_new - U) < error:
            break
        U = U_new

    #fuzzy membership matrix
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

def print_cluster_info2(clusters, centroids, feature_names, data):
    # Number of clusters
    k = len(centroids)

    print("\n===== Clustering Results =====")

    # Print info for e/a cluster
    for i in range(k):
        # Points in this cluster
        cluster_points = data[clusters == i]

        print(f"Cluster {i + 1}")
        print(f"Number of datapoints: {len(cluster_points)}")
        # print(cluster_points)

        # Print centroid coordinates
        print(f"Centroid coordinates:", end=" ")
        feature_count = len(feature_names) - 1
        for j, feature in enumerate(feature_names):
            if j < feature_count:
                print(f"{feature}: {centroids[i][j]:.4f}", end= ", ")
            elif j == feature_count:
                print(f"{feature}: {centroids[i][j]:.4f}")
        # print()

        # Print cluster datapoints
        if len(cluster_points) > 0:
            print(f"Datapoints:")
            sample_size = len(cluster_points)
            for idx in range(sample_size):
                # print(f"  Point {idx + 1}: ID={cluster_points.index[idx]}")
                # print(cluster_points.iloc[idx][["Public ID", "Gene", "IsLongevity"]].to_dict())
                public_id, gene, is_longevity = cluster_points.iloc[idx][["Public ID", "Gene", "IsLongevity"]]
                print(f" - Public ID: {public_id}, Gene: {gene}, IsLongevity: {is_longevity}")
            print()
        # print("-" * 40)

    # Print overall statistics
    print("Cluster Distribution")
    unique, counts = np.unique(clusters, return_counts=True)
    for i, count in zip(unique, counts):
        print(f"Cluster {i + 1}: {count} points ({count / len(clusters) * 100:.1f}%)")

with open("candidate_genes_list.txt", "r") as file:
    candidate_gene = [line.strip() for line in file]

with open("longevity_genes_list.txt", "r") as file:
    longevity_gene = [line.strip() for line in file]

all_genes = read_file("Longotor1delta.xls")

# Extract candidate & longevity genes if exists in full dataset.
candidate_genes = all_genes[all_genes["Public ID"].isin(candidate_gene)].copy()
longevity_genes = all_genes[all_genes["Public ID"].isin(longevity_gene)].copy()

# Add isLongevity column
candidate_genes.loc[:, "IsLongevity"] = False
longevity_genes.loc[:, "IsLongevity"] = True

# Combine dataframes
frames = [candidate_genes, longevity_genes]
combined_genes = pd.concat(frames)

# candidate_genes_normalized = z_score_norm(candidate_genes)
# longevity_genes_normalized = z_score_norm(longevity_genes)
combined_genes_normalized = z_score_norm(combined_genes)

for k in range(3, 11, 2):
    print(f"Running K-Means Clustering with k={k}")
    clusters, centroids = k_means_cluster(combined_genes_normalized, k=k, q=1, max_iterations=100)
    feature_names = ['sch9/wt', 'ras2/wt', 'tor1/wt']
    classes = ["Cluster 1", "Cluster 2", "Cluster 3"]
    print_cluster_info2(clusters, centroids, feature_names, combined_genes_normalized)
    print("-" * 75)
centers, U, labels = fuzzy_c_means(combined_genes_normalized[['sch9/wt', 'ras2/wt', 'tor1/wt']].values, n_clusters=3)
print(U)

for k in range(3, 11, 2):
    print(f"Running Fuzzy C-Means Clustering with k={k}")
    X = combined_genes_normalized[['sch9/wt', 'ras2/wt', 'tor1/wt']].values
    centers, U, labels = fuzzy_c_means(X, n_clusters=k)
    feature_names = ['sch9/wt', 'ras2/wt', 'tor1/wt']
    # Print cluster info
    print_cluster_info2(labels, centers, feature_names, combined_genes_normalized)
    # Print the fuzzy membership matrix
    print("\n===== Fuzzy Membership Matrix =====")
    membership_df = pd.DataFrame(U.T, columns=[f"Cluster {i + 1}" for i in range(k)])
    membership_df['Public ID'] = combined_genes_normalized['Public ID'].values
    membership_df['Gene'] = combined_genes_normalized['Gene'].values
    membership_df['IsLongevity'] = combined_genes_normalized['IsLongevity'].values
    print(membership_df.to_string(index=False, float_format="%.4f"))
    print("-" * 75)
