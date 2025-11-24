# Required for clustering_author function:
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
# Required for analyze_space_distance_preservation
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.stats import pearsonr, ConstantInputWarning
from typing import List, Dict, Any
from tabulate import tabulate

import json 

def sample_ds(input_file, output_file, num_insts=10000, min_num_text_per_inst=0, max_num_text_per_inst=3):

    """
    Usage 
    sample_ds('/mnt/swordfish-pool2/nikhil/raw_all/data.jsonl', '/mnt/swordfish-pool2/milad/hiatus-data/reddit_cluster_training.pkl', 
          num_insts=5000, 
          min_num_text_per_inst=3, 
          max_num_text_per_inst=10)
    """
    f = open(input_file)
    out_list = []
    for i in range(num_insts):    
        json_obj = json.loads(f.readline())
        out_list.append({
            'fullText': json_obj['syms'],
            'authorID': json_obj['author_id']
        })
    df = pd.DataFrame(out_list)
    df.to_pickle(output_file)

def _calculate_silhouette_score(X: np.ndarray, labels: np.ndarray, metric: str) -> float | None:
    """
    Calculates the silhouette score for a given clustering result.

    Args:
        X (np.ndarray): The input data (embeddings).
        labels (np.ndarray): The cluster labels for each point in X.
        metric (str): The distance metric used for the score calculation.

    Returns:
        float | None: The silhouette score, or None if it cannot be computed.
    """
    unique_labels_set = set(labels)
    n_clusters_ = len(unique_labels_set) - (1 if -1 in unique_labels_set else 0)

    # The silhouette score is only defined if there is more than 1 cluster.
    # Outliers (label -1) are excluded from the score calculation.
    if n_clusters_ > 1:
        # Create a mask to select only points that are part of a cluster (not noise)
        clustered_mask = (labels != -1)
        if np.sum(clustered_mask) > 1:
            X_clustered = X[clustered_mask]
            labels_clustered = labels[clustered_mask]
            try:
                # Compute the score on the non-outlier points
                return silhouette_score(X_clustered, labels_clustered, metric=metric)
            except ValueError:
                return None
    return None


def clustering_author(background_corpus_df: pd.DataFrame,
                      test_corpus_df: pd.DataFrame = None,
                      embedding_clm: str = 'style_embedding',
                      eps_values: List[float] = None,
                      min_samples: int = 5,
                      pca_dimensions: int | None = None,
                      metric: str = 'cosine') -> pd.DataFrame:
    """
    Performs DBSCAN clustering on embeddings in a DataFrame.

    Experiments with different `eps` parameters to find a clustering
    that maximizes the silhouette score, indicating well-separated clusters.

    Args:
        background_corpus_df (pd.DataFrame): DataFrame with an embedding column.
        embedding_clm (str): Name of the column containing embeddings.
                             Each embedding should be a list or NumPy array.
        eps_values (List[float], optional): Specific `eps` values to test.
                                            If None, a default range is used.
                                            For 'cosine' metric, eps is typically in [0, 2].
                                            For 'euclidean', scale depends on embedding magnitudes.
        min_samples (int): DBSCAN `min_samples` parameter. Minimum number of
                           samples in a neighborhood for a point to be a core point.
        pca_dimensions (int | None): If an integer is provided, PCA will be applied to reduce
                                     embeddings to this number of dimensions before clustering.
        metric (str): The distance metric to use for DBSCAN and silhouette score
                      (e.g., 'cosine', 'euclidean').

    Returns:
        pd.DataFrame: The input DataFrame with a new 'cluster_label' column.
                      Labels are from the DBSCAN run with the highest silhouette score.
                      If no suitable clustering is found, labels might be all -1 (noise).
    """
    if embedding_clm not in background_corpus_df.columns:
        raise ValueError(f"Embedding column '{embedding_clm}' not found in DataFrame.")

    embeddings_list = background_corpus_df[embedding_clm].tolist()
    

    X_list = []
    original_indices = [] # To map results back to the original DataFrame's indices
    
    for i, emb_val in enumerate(embeddings_list):
        if emb_val is not None:
            try:
                e = np.asarray(emb_val, dtype=float)
                if e.ndim == 1 and e.size > 0: # Standard 1D vector
                    X_list.append(e)
                    original_indices.append(i)
                elif e.ndim == 0 and e.size == 1: # Scalar value, treat as 1D vector of size 1
                    X_list.append(np.array([e.item()]))
                    original_indices.append(i)
                # Silently skip empty arrays or improperly shaped arrays
            except (TypeError, ValueError):
                # Silently skip if conversion to float array fails
                pass
        
    # Initialize labels for all rows in the original DataFrame to -1 (noise/unprocessed)
    final_labels_for_df = pd.Series(-1, index=background_corpus_df.index, dtype=int)

    if not X_list:
        print(f"No valid embeddings found in column '{embedding_clm}'. Assigning all 'cluster_label' as -1.")
        background_corpus_df['cluster_label'] = final_labels_for_df
        return background_corpus_df

    X = np.array(X_list) # Creates a 2D array from the list of 1D arrays
    original_embeddings_list = [embeddings_list[i] for i in original_indices]

    if X.shape[0] == 1:
        print("Only one valid embedding found. Assigning cluster label 0 to it.")
        if original_indices: # Should always be true if X.shape[0]==1 from X_list
            final_labels_for_df.iloc[original_indices[0]] = 0
        background_corpus_df['cluster_label'] = final_labels_for_df
        return background_corpus_df

    if X.shape[0] < min_samples:
        print(f"Number of valid embeddings ({X.shape[0]}) is less than min_samples ({min_samples}). "
              f"All valid embeddings will be marked as noise (-1).")
        for original_idx in original_indices:
             final_labels_for_df.iloc[original_idx] = -1
        background_corpus_df['cluster_label'] = final_labels_for_df
        return background_corpus_df

    # --- Optional: Apply PCA for dimensionality reduction ---
    if pca_dimensions is not None and X.shape[1] > pca_dimensions:
        from sklearn.decomposition import PCA
        print(f"Applying PCA to reduce dimensions from {X.shape[1]} to {pca_dimensions}...")
        pca = PCA(n_components=pca_dimensions, random_state=42)
        X = pca.fit_transform(X)
        
        # Update the background_corpus_df with the transformed embeddings
        # This ensures subsequent centroid calculations use the reduced-dimension space.
        background_corpus_df[embedding_clm] = list(X)

        # If a test set is provided, transform its embeddings using the same PCA model
        if test_corpus_df is not None:
            test_embeddings_matrix = _safe_embeddings_to_matrix(test_corpus_df[embedding_clm])
            if test_embeddings_matrix.ndim == 2 and test_embeddings_matrix.shape[0] > 0 and test_embeddings_matrix.shape[1] == pca.n_features_in_:
                print(f"Transforming test set embeddings with the same PCA model...")
                transformed_test_embeddings = pca.transform(test_embeddings_matrix)
                # Update the test DataFrame's embedding column with the reduced embeddings
                #test_corpus_df.loc[:, embedding_clm] = list(transformed_test_embeddings)
                test_corpus_df[embedding_clm] = list(transformed_test_embeddings)
            else:
                print(f"Warning: Could not apply PCA to test set. Test shape: {test_embeddings_matrix.shape}, PCA features: {pca.n_features_in_}")


    # For cosine metric, normalize embeddings to unit length.
    # This is standard practice as cosine similarity is equivalent to Euclidean
    # distance on L2-normalized vectors. DBSCAN's 'cosine' metric internally
    # works with these normalized distances.
    if metric == 'cosine':
        from sklearn.preprocessing import normalize
        print("Normalizing embeddings for cosine distance...")
        X_normalized = normalize(X, norm='l2', axis=1)
        # Update the background_corpus_df with the normalized embeddings
        background_corpus_df[embedding_clm] = list(X_normalized)
        X = X_normalized # Use the normalized data for clustering

        # Also normalize the test corpus embeddings if they exist
        if test_corpus_df is not None:
            print("Normalizing test corpus embeddings for cosine distance...")
            test_embeddings_matrix = _safe_embeddings_to_matrix(test_corpus_df[embedding_clm])
            if test_embeddings_matrix.ndim == 2 and test_embeddings_matrix.shape[0] > 0:
                normalized_test_embeddings = normalize(test_embeddings_matrix, norm='l2', axis=1)
                test_corpus_df[embedding_clm] = list(normalized_test_embeddings)
            else:
                print("Warning: Could not normalize test set embeddings due to invalid data.")
    
    if eps_values is None:
        if metric == 'cosine':
            #eps_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            eps_values = np.arange(0.01, 0.2, 0.01)
        else: # 'euclidean' or other
            if X.shape[0] > 1:
                # For Euclidean, eps depends on the scale of the data.
                # A simple heuristic: a fraction of the data's standard deviation.
                data_spread = np.std(X)
                eps_values = [round(data_spread * f, 2) for f in [0.25, 0.5, 1.0]]
                eps_values = [e for e in eps_values if e > 1e-6] # Filter out zero or near-zero eps
            if not eps_values or X.shape[0] <=1: # Fallback if heuristic fails or not enough data
                 eps_values = [0.5, 1.0, 1.5] 
        print(f"Warning: `eps_values` not provided. Using default range for metric '{metric}': {eps_values}. "
              f"It's recommended to supply `eps_values` tuned to your data.")

    print(f"\n--- Starting DBSCAN Clustering & Evaluation ---")
    print(f"Metric: '{metric}', Min Samples: {min_samples}, EPS values: {[f'{e:.2f}' for e in eps_values]}")

    best_score = -1.001
    best_labels = None
    best_eps = None
    results_for_table = []

    # This loop now lives in `clustering_author` to have access to the full DataFrame for evaluation.
    for eps in eps_values:
        if eps <= 1e-9: continue

        print(f"\nTesting eps = {eps:.3f}...")
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        current_labels = db.fit_predict(X)

        # --- Evaluation Step 1: Silhouette Score ---
        num_clusters = len(set(current_labels) - {-1})
        num_outliers = np.sum(current_labels == -1)
        score = _calculate_silhouette_score(X, current_labels, metric)
        if score is not None:
            print(f"  - Silhouette Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_labels = current_labels.copy()
                best_eps = eps
        else:
            print("  - Silhouette Score: N/A (not enough clusters found)")

        # --- Evaluation Step 2: Distance Preservation ---
        # Temporarily assign labels to a copy of the DataFrame for evaluation
        temp_df = background_corpus_df.copy()
        temp_labels_for_df = pd.Series(-1, index=temp_df.index, dtype=int)
        temp_labels_for_df.iloc[original_indices] = current_labels
        temp_df['cluster_label'] = temp_labels_for_df

        correlation = analyze_space_distance_preservation(temp_df, embedding_clm, 'cluster_label')
        if correlation is not None:
            print(f"  - Distance Preservation (Pearson r): {correlation:.4f}")
        else:
            print("  - Distance Preservation (Pearson r): N/A (not enough clusters/data)")

        # --- Evaluation Step 3: Distance Preservation on Test Corpus (if provided) ---
        if test_corpus_df is not None:
            test_correlation = None
            # We need the centroids from the current clustering of the background corpus
            centroids = _compute_cluster_centroids(temp_df[temp_df['cluster_label'] != -1], embedding_clm, 'cluster_label')
            test_correlation = evaluate_test_set_distance_preservation(test_corpus_df, centroids, embedding_clm)
            if test_correlation is not None:
                print(f"  - Test Set Distance Preservation (Pearson r): {test_correlation:.4f}")
            else:
                print("  - Test Set Distance Preservation (Pearson r): N/A (not enough test data or clusters)")
        
        print('Eps {}, #clusters {}, solihouette {}, Pearson {}'.format(eps, len(set(current_labels) - {-1}), score, test_correlation))
        results_for_table.append([f"{eps:.3f}", f"{score:.4f}" if score is not None else "N/A", f"{test_correlation:.4f}" if test_correlation is not None else "N/A", num_clusters, num_outliers])

    # --- Print Final Summary Table ---
    print("\n\n--- Clustering Run Summary ---")
    headers = ["Epsilon (eps)", "Silhouette Score", "Test Dist. Preserv.", "# Clusters", "# Outliers"]
    print(tabulate(results_for_table, headers=headers, tablefmt="grid"))
    print("----------------------------\n")
    
    if best_labels is not None:
        num_found_clusters = len(set(best_labels) - {-1})
        print(f"\n--- Best Clustering Result ---")
        print(f"Best eps: {best_eps:.3f} yielded the highest Silhouette Score: {best_score:.4f} ({num_found_clusters} clusters).")
        for i, label in enumerate(best_labels): 
            original_df_idx = original_indices[i] 
            final_labels_for_df.iloc[original_df_idx] = label
    else:
        print("No suitable DBSCAN clustering found meeting criteria. All processed embeddings marked as noise (-1).")

    background_corpus_df['cluster_label'] = final_labels_for_df
    # restore the original embedding
    print(original_embeddings_list[0].shape)
    background_corpus_df[embedding_clm] = original_embeddings_list
    return background_corpus_df


def _safe_embeddings_to_matrix(embeddings_column: pd.Series) -> np.ndarray:
    """
    Converts a pandas Series of embeddings (expected to be lists of floats or 1D np.arrays)
    into a 2D NumPy matrix. Handles None values and attempts to stack consistently.
    Returns an empty 2D array (e.g., shape (0,0) or (0,D)) if conversion fails or no valid data.
    """
    embeddings_list = embeddings_column.tolist()
    
    processed_1d_arrays = []
    for emb in embeddings_list:
        if emb is not None:
            if hasattr(emb, '__iter__') and not isinstance(emb, (str, bytes)):
                try:
                    arr = np.asarray(emb, dtype=float)
                    if arr.ndim == 1 and arr.size > 0:
                        processed_1d_arrays.append(arr)
                except (TypeError, ValueError):
                    pass # Ignore embeddings that cannot be converted

    if not processed_1d_arrays:
        return np.empty((0,0))

    # Check for consistent dimensionality before vstacking
    first_len = processed_1d_arrays[0].shape[0]
    consistent_embeddings = [arr for arr in processed_1d_arrays if arr.shape[0] == first_len]

    if not consistent_embeddings:
        return np.empty((0, first_len if processed_1d_arrays else 0)) # (0,D) or (0,0)

    try:
        return np.vstack(consistent_embeddings)
    except ValueError:
        # Should not happen if lengths are consistent
        return np.empty((0, first_len))


def _compute_cluster_centroids(
    df_clustered_items: pd.DataFrame, # DataFrame already filtered for non-noise items
    embedding_clm: str,
    cluster_label_clm: str
) -> Dict[Any, np.ndarray]:
    """Computes the centroid for each cluster from a pre-filtered DataFrame."""
    centroids = {}
    if df_clustered_items.empty:
        return centroids

    for cluster_id, group in df_clustered_items.groupby(cluster_label_clm):
        embeddings_matrix = _safe_embeddings_to_matrix(group[embedding_clm])
        
        if embeddings_matrix.ndim == 2 and embeddings_matrix.shape[0] > 0 and embeddings_matrix.shape[1] > 0:
            centroids[cluster_id] = np.mean(embeddings_matrix, axis=0)
    return centroids


def _project_to_centroid_space(
    original_embeddings_matrix: np.ndarray, # (n_items, n_original_features)
    centroids_map: Dict[Any, np.ndarray]    # {cluster_id: centroid_vector (n_original_features,)}
) -> np.ndarray:
    """Projects embeddings into a new space defined by cluster centroids using cosine similarity."""
    if not centroids_map or original_embeddings_matrix.ndim != 2 or \
       original_embeddings_matrix.shape[0] == 0 or original_embeddings_matrix.shape[1] == 0:
        return np.empty((original_embeddings_matrix.shape[0], 0)) # (n_items, 0_new_features)

    sorted_cluster_ids = sorted(centroids_map.keys())
    
    valid_centroid_vectors = []
    for cid in sorted_cluster_ids:
        centroid_vec = centroids_map[cid]
        if isinstance(centroid_vec, np.ndarray) and centroid_vec.ndim == 1 and \
           centroid_vec.size == original_embeddings_matrix.shape[1]:
            valid_centroid_vectors.append(centroid_vec)

    if not valid_centroid_vectors:
        return np.empty((original_embeddings_matrix.shape[0], 0))

    centroid_matrix = np.vstack(valid_centroid_vectors) # (n_valid_centroids, n_original_features)
    
    # Result: (n_items, n_valid_centroids)
    projected_matrix = cosine_similarity(original_embeddings_matrix, centroid_matrix)
    return projected_matrix


def _get_pairwise_cosine_distances(embeddings_matrix: np.ndarray) -> np.ndarray:
    """Calculates unique pairwise cosine distances from an embedding matrix."""
    if not isinstance(embeddings_matrix, np.ndarray) or embeddings_matrix.ndim != 2 or \
       embeddings_matrix.shape[0] < 2 or embeddings_matrix.shape[1] == 0:
        return np.array([]) # Not enough samples or features
        
    dist_matrix = cosine_distances(embeddings_matrix)
    iu = np.triu_indices(dist_matrix.shape[0], k=1) # Upper triangle, excluding diagonal
    return dist_matrix[iu]


def analyze_space_distance_preservation(
    df: pd.DataFrame,
    embedding_clm: str = 'style_embedding',
    cluster_label_clm: str = 'cluster_label'
) -> float | None:
    """
    Analyzes how well a new space, defined by cluster centroids, preserves
    the cosine distance relationships from the original embedding space.

    Args:
        df (pd.DataFrame): DataFrame with original embeddings and cluster labels.
        embedding_clm (str): Column name for original embeddings.
        cluster_label_clm (str): Column name for cluster labels.

    Returns:
        float | None: Pearson correlation coefficient. Returns None if analysis
                      cannot be performed (e.g., <2 clusters, <2 items), or 0.0
                      if correlation is NaN (e.g. due to zero variance in distances).
    """
    df_valid_items = df[df[cluster_label_clm] != -1].copy()

    if df_valid_items.shape[0] < 2:
        return None # Need at least 2 items for pairwise distances

    original_embeddings_matrix = _safe_embeddings_to_matrix(df_valid_items[embedding_clm])
    if original_embeddings_matrix.ndim != 2 or original_embeddings_matrix.shape[0] < 2 or \
       original_embeddings_matrix.shape[1] == 0:
        return None # Valid matrix from original embeddings could not be formed

    centroids = _compute_cluster_centroids(df_valid_items, embedding_clm, cluster_label_clm)
    if len(centroids) < 2: # Need at least 2 centroids for a multi-dimensional new space
        return None

    projected_embeddings_matrix = _project_to_centroid_space(original_embeddings_matrix, centroids)
    if projected_embeddings_matrix.ndim != 2 or projected_embeddings_matrix.shape[0] < 2 or \
       projected_embeddings_matrix.shape[1] < 2: # New space needs at least 2 dimensions (centroids)
        return None

    distances_original_space = _get_pairwise_cosine_distances(original_embeddings_matrix)
    distances_new_space = _get_pairwise_cosine_distances(projected_embeddings_matrix)

    if distances_original_space.size == 0 or distances_new_space.size == 0 or \
       distances_original_space.size != distances_new_space.size:
        return None # Mismatch or empty distances

    try:
        # Catching ConstantInputWarning that pearsonr can raise
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConstantInputWarning)
            correlation, _ = pearsonr(distances_original_space, distances_new_space)
    except (ValueError, ConstantInputWarning):
        # This happens if one of the distance arrays has zero variance (all distances are the same).
        # This is a valid case where correlation is undefined or 0.
        return 0.0
    except Exception: # Safeguard for other unexpected errors
        return None

    if np.isnan(correlation):
        return 0.0 # Default for NaN correlation

    return correlation

def evaluate_test_set_distance_preservation(
    test_df: pd.DataFrame,
    centroids_map: Dict[Any, np.ndarray],
    embedding_clm: str = 'style_embedding'
) -> float | None:
    """
    Evaluates how well a centroid space (from a background corpus) preserves
    distances for a separate test corpus.

    Args:
        test_df (pd.DataFrame): The test corpus DataFrame with embeddings.
        centroids_map (Dict[Any, np.ndarray]): A map of cluster IDs to centroid vectors,
                                               pre-computed from the background corpus.
        embedding_clm (str): The name of the embedding column.

    Returns:
        float | None: Pearson correlation coefficient, or None if analysis is not possible.
    """
    if test_df.shape[0] < 2:
        return None # Need at least 2 items for pairwise distances

    if not centroids_map or len(centroids_map) < 2:
        return None # Need at least 2 centroids to define a meaningful projected space

    # 1. Get original embeddings and distances for the test set
    test_embeddings_matrix = _safe_embeddings_to_matrix(test_df[embedding_clm])
    if test_embeddings_matrix.ndim != 2 or test_embeddings_matrix.shape[0] < 2:
        return None # Not enough valid embeddings in the test set

    distances_original_space = _get_pairwise_cosine_distances(test_embeddings_matrix)

    # 2. Project test embeddings into the centroid space and get new distances
    projected_embeddings_matrix = _project_to_centroid_space(test_embeddings_matrix, centroids_map)


    if projected_embeddings_matrix.ndim != 2 or projected_embeddings_matrix.shape[1] < 2:
        return None # Projection failed or resulted in a space with <2 dimensions

    distances_new_space = _get_pairwise_cosine_distances(projected_embeddings_matrix)

    # 3. Calculate Pearson correlation
    if distances_original_space.size != distances_new_space.size or distances_original_space.size == 0:
        return None

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConstantInputWarning)
            correlation, _ = pearsonr(distances_original_space, distances_new_space)
    except (ValueError, ConstantInputWarning):
        return 0.0 # Zero variance in one of the distance sets

    return correlation if not np.isnan(correlation) else 0.0