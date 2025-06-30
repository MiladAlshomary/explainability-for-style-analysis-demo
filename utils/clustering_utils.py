# Required for clustering_author function:
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
# Required for analyze_space_distance_preservation
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.stats import pearsonr
from typing import List, Dict, Any

def _find_best_dbscan_eps(X: np.ndarray,
                          eps_values: List[float],
                          min_samples: int,
                          metric: str) -> tuple[float | None, np.ndarray | None, float]:
    """
    Iterates through eps_values for DBSCAN and returns the parameters
    that yield the highest silhouette score.

    Args:
        X (np.ndarray): The input data (embeddings).
        eps_values (List[float]): List of eps values to try.
        min_samples (int): DBSCAN min_samples parameter.
        metric (str): Distance metric for DBSCAN and silhouette score.

    Returns:
        tuple[float | None, np.ndarray | None, float]:
            - best_eps: The eps value that resulted in the best score. None if no suitable clustering.
            - best_labels: The cluster labels from the best DBSCAN run. None if no suitable clustering.
            - best_score: The highest silhouette score achieved.
    """
    best_score = -1.001  # Silhouette score is in [-1, 1]
    best_labels = None
    best_eps = None

    for eps in eps_values:
        if eps <= 1e-9:  # eps must be positive
            continue
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = db.fit_predict(X)

        unique_labels_set = set(labels)
        n_clusters_ = len(unique_labels_set) - (1 if -1 in unique_labels_set else 0)

        if n_clusters_ > 1:
            clustered_mask = (labels != -1)
            if np.sum(clustered_mask) >= 2:  # Need at least 2 non-noise points
                X_clustered = X[clustered_mask]
                labels_clustered = labels[clustered_mask]
                try:
                    score = silhouette_score(X_clustered, labels_clustered, metric=metric)
                    if score > best_score:
                        best_score = score
                        best_labels = labels.copy()
                        best_eps = eps
                except ValueError:  # Catch errors from silhouette_score
                    pass
        elif n_clusters_ == 1 and best_labels is None: # Fallback for single cluster
            if not all(l == -1 for l in labels):
                current_score_for_single_cluster = -0.5 # Nominal score
                if current_score_for_single_cluster > best_score:
                    best_score = current_score_for_single_cluster
                    best_labels = labels.copy()
                    best_eps = eps
    return best_eps, best_labels, best_score

def clustering_author(background_corpus_df: pd.DataFrame,
                      embedding_clm: str = 'style_embedding',
                      eps_values: List[float] = None,
                      min_samples: int = 5,
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

    if eps_values is None:
        if metric == 'cosine':
            eps_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        else: 
            if X.shape[0] > 1:
                data_spread = np.std(X) 
                eps_values = [round(data_spread * f, 2) for f in [0.25, 0.5, 1.0]]
                eps_values = [e for e in eps_values if e > 1e-6] 
            if not eps_values or X.shape[0] <=1: 
                 eps_values = [0.5, 1.0, 1.5] 
        print(f"Warning: `eps_values` not provided. Using default range for metric '{metric}': {eps_values}. "
              f"It's recommended to supply `eps_values` tuned to your data.")

    print(f"Performing DBSCAN clustering (min_samples={min_samples}, metric='{metric}') with eps values: "
          f"{[f'{e:.2f}' for e in eps_values]}")

    best_eps, best_labels, best_score = _find_best_dbscan_eps(X, eps_values, min_samples, metric)

    if best_labels is not None:
        num_found_clusters = len(set(best_labels) - {-1})
        print(f"Best clustering found: eps={best_eps:.2f}, Silhouette Score={best_score:.4f} ({num_found_clusters} clusters).")
        for i, label in enumerate(best_labels): 
            original_df_idx = original_indices[i] 
            final_labels_for_df.iloc[original_df_idx] = label
    else:
        print("No suitable DBSCAN clustering found meeting criteria. All processed embeddings marked as noise (-1).")

    background_corpus_df['cluster_label'] = final_labels_for_df
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

    # Handle cases where variance is zero in one of the distance arrays (leads to NaN correlation)
    if np.all(distances_new_space == distances_new_space[0]) or \
       np.all(distances_original_space == distances_original_space[0]):
        return 0.0 # Correlation is undefined or 0 if one variable is constant

    try:
        correlation, _ = pearsonr(distances_original_space, distances_new_space)
    except ValueError: # Should be caught by variance checks, but as a safeguard
        return None

    if np.isnan(correlation):
        return 0.0 # Default for NaN correlation
        
    return correlation