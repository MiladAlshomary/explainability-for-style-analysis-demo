import sys

import pandas as pd
import numpy as np
import math
from collections import Counter, defaultdict
from typing import List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
import hashlib
import json
from gram2vec import vectorizer
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

CACHE_DIR = "datasets/embeddings_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
# Bump this whenever there is a change etc...
CACHE_VERSION = 1

class style_analysis_schema(BaseModel):
    features: list[str]
    spans: dict[str, dict[str, list[str]]]


def compute_g2v_features(clustered_authors_df: pd.DataFrame, task_authors_df: pd.DataFrame=None, text_clm='fullText') -> pd.DataFrame:
    """
    Computes gram2vec feature vectors for each author and adds them to the DataFrame.
    This effectively creates a mapping from each author to their vector.
    """
    if task_authors_df is not None:
        clustered_authors_df = pd.concat([task_authors_df, clustered_authors_df])

    # Gather the input texts (preserves list-of-strings if any)
    #texts = background_corpus_df[text_clm].fillna("").tolist()
    author_texts = ['\n\n'.join(x) for x in clustered_authors_df.fullText.tolist()]

    # Create a reproducible JSON serialization of the texts
    serialized = json.dumps({
        "col": text_clm,
        "texts": author_texts
    }, sort_keys=True, ensure_ascii=False)

    # Compute MD5 hash
    digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{digest}.pkl")

    # If cache hit, load and return
    if os.path.exists(cache_path):
        print(f"Cache hit...")
        with open(cache_path, "rb") as f:
            clustered_authors_df = pickle.load(f)
    
    else: # Else compute and cache
        g2v_feats_df = vectorizer.from_documents(author_texts, batch_size=16)
        author_to_g2v_feats = {x[0]: x[1] for x in zip(clustered_authors_df.authorID.tolist(), g2v_feats_df.to_numpy().tolist())}

        # apply normalization
        vector_std  = np.std(list(author_to_g2v_feats.values()), axis=0)
        vector_mean = np.mean(list(author_to_g2v_feats.values()), axis=0)
        vector_std[vector_std == 0] = 1.0
        author_to_g2v_feats_z_normalized = {x[0]: (x[1] - vector_mean) / vector_std for x in author_to_g2v_feats.items()}
        

        # Add the vectors as a new column of the DataFrame.
        clustered_authors_df['g2v_vector'] = [{x[1]: x[0] for x in zip(val, g2v_feats_df.columns.tolist())} 
                                            for val in author_to_g2v_feats_z_normalized.values()]
        
        with open(cache_path, "wb") as f:
            pickle.dump(clustered_authors_df, f)
    
    if task_authors_df is not None:
        task_authors_df = clustered_authors_df[clustered_authors_df.authorID.isin(task_authors_df.authorID.tolist())]
        clustered_authors_df = clustered_authors_df[~clustered_authors_df.authorID.isin(task_authors_df.authorID.tolist())]


    return clustered_authors_df, task_authors_df


def get_task_authors_from_background_df(background_df):
    task_authors_df = background_df[background_df.authorID.isin(["Q_author", "a0_author", "a1_author", "a2_author"])]
    return task_authors_df

def instance_to_df(instance, predicted_author=None, ground_truth_author=None):
    #create a dataframe of the task authors
    task_authos_df  = pd.DataFrame([
        {'authorID': 'Mystery author', 'fullText': instance['Q_fullText'], 'predicted': None, 'ground_truth': None},
        {'authorID': 'Candidate Author 1', 'fullText': instance['a0_fullText'], 'predicted': predicted_author == 0, 'ground_truth': ground_truth_author == 0},
        {'authorID': 'Candidate Author 2', 'fullText': instance['a1_fullText'], 'predicted': predicted_author == 1, 'ground_truth': ground_truth_author == 1},
        {'authorID': 'Candidate Author 3', 'fullText': instance['a2_fullText'], 'predicted': predicted_author == 2, 'ground_truth': ground_truth_author == 2}
                    
    ])

    if type(instance['Q_fullText']) == list:
        task_authos_df = task_authos_df.groupby('authorID').agg({'fullText': lambda x: list(x)}).reset_index()

    return task_authos_df


def generate_style_embedding(background_corpus_df: pd.DataFrame, text_clm: str, model_name: str) -> pd.DataFrame:
    """
    Generates style embeddings for documents in a background corpus using a specified model.
    If a row in `text_clm` contains a list of strings, the final embedding for that row
    is the average of the embeddings of all strings in the list.

    Args:
        background_corpus_df (pd.DataFrame): DataFrame containing the corpus.
        text_clm (str): Name of the column containing the text data (either string or list of strings).
        model_name (str): Name of the model to use for generating embeddings.

    Returns:
        pd.DataFrame: The input DataFrame with a new column for style embeddings.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    if model_name not in [
        'gabrielloiseau/LUAR-MUD-sentence-transformers',
        'gabrielloiseau/LUAR-CRUD-sentence-transformers',
        'miladalsh/light-luar',
        'AnnaWegmann/Style-Embedding',

    ]:
        print('Model is not supported')
        return background_corpus_df
    
    print(f"Generating style embeddings using {model_name} on column '{text_clm}'...")

    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()

    # Heuristic to check if the column contains lists of strings by checking the first valid item.
    # This assumes the column is homogenous.
    is_list_column = False
    if not background_corpus_df.empty:
        # Get the first non-NaN value to inspect its type
        series_no_na = background_corpus_df[text_clm].dropna()
        if not series_no_na.empty:
            first_valid_item = series_no_na.iloc[0]
            if isinstance(first_valid_item, list):
                is_list_column = True

    if is_list_column:
        # Flatten all texts into a single list for batch processing
        texts_to_encode = []
        row_lengths = []
        for text_list in background_corpus_df[text_clm]:
            # Ensure we handle None, empty lists, or other non-list types gracefully
            if isinstance(text_list, list) and text_list:
                texts_to_encode.extend(text_list)
                row_lengths.append(len(text_list))
            else:
                row_lengths.append(0)

        if texts_to_encode:
            all_embeddings = model.encode(texts_to_encode, convert_to_tensor=True, show_progress_bar=True)
        else:
            all_embeddings = torch.empty((0, embedding_dim), device=model.device)

        # Reconstruct and average embeddings for each row
        final_embeddings = []
        current_pos = 0
        for length in row_lengths:
            if length > 0:
                row_embeddings = all_embeddings[current_pos:current_pos + length]
                avg_embedding = torch.mean(row_embeddings, dim=0)
                final_embeddings.append(avg_embedding.cpu().numpy())
                current_pos += length
            else:
                final_embeddings.append(np.zeros(embedding_dim))
    else:
        # Column contains single strings
        texts = background_corpus_df[text_clm].fillna("").tolist()
        # convert_to_tensor=False is faster if we just need numpy arrays
        embeddings = model.encode(texts, show_progress_bar=True)
        final_embeddings = list(embeddings)

    # Create a clean column name from the model name
    col_name = f'{model_name.split("/")[-1]}_style_embedding'
    background_corpus_df[col_name] = final_embeddings

    return background_corpus_df

# ── wrapper with caching ───────────────────────────────────────
def cached_generate_style_embedding(background_corpus_df: pd.DataFrame,
                                    text_clm: str,
                                    model_name: str) -> pd.DataFrame:
    """
    Wraps `generate_style_embedding`, caching its output in pickle files
    keyed by an MD5 of (model_name + text list). If the cache exists,
    loads and returns it instead of recomputing.
    """

    # Gather the input texts (preserves list-of-strings if any)
    texts = background_corpus_df[text_clm].fillna("").tolist()

    # Create a reproducible JSON serialization of the texts
    serialized = json.dumps({
        "model": model_name,
        "col": text_clm,
        "texts": texts
    }, sort_keys=True, ensure_ascii=False)

    # Compute MD5 hash
    digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{digest}.pkl")

    # If cache hit, load and return
    if os.path.exists(cache_path):
        print(f"Cache hit for {model_name} on column '{text_clm}'")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Otherwise, compute, cache, and return
    df_with_emb = generate_style_embedding(background_corpus_df, text_clm, model_name)
    print(f"Computing embeddings for {model_name} on column '{text_clm}', saving to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(df_with_emb, f)
    return df_with_emb

def get_style_feats_distribution(documentIDs, style_feats_dict):
    style_feats = []
    for documentId in documentIDs:
        if documentId not in document_to_style_feats:
            #print(documentId)
            continue

        style_feats+= document_to_style_feats[documentId]

    tfidf = [style_feats.count(key) * val for key, val in style_feats_dict.items()]

    return tfidf

def get_cluster_top_feats(style_feats_distribution, style_feats_list, top_k=5):
    sorted_feats = np.argsort(style_feats_distribution)[::-1]
    top_feats = [style_feats_list[x] for x in sorted_feats[:top_k] if style_feats_distribution[x] > 0]
    return top_feats

def compute_clusters_style_representation(
    background_corpus_df: pd.DataFrame,
    cluster_ids: List[Any],
    other_cluster_ids: List[Any],
    features_clm_name: str,
    cluster_label_clm_name: str = 'cluster_label',
    top_n: int = 10
) -> List[str]:
    """
    Given a DataFrame with document IDs, cluster IDs, and feature lists,
    return the top N features that are most important in the specified `cluster_ids`
    while having low importance in `other_cluster_ids`.
    Importance is determined by TF-IDF scores. The final score for a feature is
    (summed TF-IDF in `cluster_ids`) - (summed TF-IDF in `other_cluster_ids`).

    Parameters:
    - background_corpus_df: pd.DataFrame. Must contain the columns specified by
                            `cluster_label_clm_name` and `features_clm_name`.
                            The column `features_clm_name` should contain lists of strings (features).
    - cluster_ids: List of cluster IDs for which to find representative features (target clusters).
    - other_cluster_ids: List of cluster IDs whose features should be down-weighted.
                         Features prominent in these clusters will have their scores reduced.
                         Pass an empty list or None if no contrastive clusters are needed.
    - features_clm_name: The name of the column in `background_corpus_df` that
                         contains the list of features for each document.
    - cluster_label_clm_name: The name of the column in `background_corpus_df`
                              that contains the cluster labels. Defaults to 'cluster_label'.
    - top_n: Number of top features to return.
    Returns:
    - List[str]: A list of feature names. These are up to `top_n` features
                 ranked by their adjusted TF-IDF scores (score in `cluster_ids`
                 minus score in `other_cluster_ids`). Only features with a final
                 adjusted score > 0 are included.
    """

    assert background_corpus_df[features_clm_name].apply(
        lambda x: isinstance(x, list) and all(isinstance(feat, str) for feat in x)
    ).all(), f"Column '{features_clm_name}' must contain lists of strings."

    # Compute TF-IDF on the entire corpus
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None  # Disable default token pattern, treat items in list as tokens
    )
    tfidf_matrix = vectorizer.fit_transform(background_corpus_df[features_clm_name])
    feature_names = vectorizer.get_feature_names_out()

    # Get boolean mask for documents in selected clusters
    selected_mask = background_corpus_df[cluster_label_clm_name].isin(cluster_ids).to_numpy()

    if not selected_mask.any():
        return [] # No documents found for the given cluster_ids

    # Subset the TF-IDF matrix using the boolean mask
    selected_tfidf = tfidf_matrix[selected_mask]

    # Sum TF-IDF scores across documents for each feature in the target clusters
    target_feature_scores_sum = selected_tfidf.sum(axis=0).A1  # Convert to 1D array

    # Initialize adjusted scores with target scores
    adjusted_feature_scores = target_feature_scores_sum.copy()

    # If other_cluster_ids are provided and not empty, subtract their TF-IDF sums
    if other_cluster_ids: # Checks if the list is not None and not empty
        other_selected_mask = background_corpus_df[cluster_label_clm_name].isin(other_cluster_ids).to_numpy()

        if other_selected_mask.any():
            other_selected_tfidf = tfidf_matrix[other_selected_mask]
            contrast_feature_scores_sum = other_selected_tfidf.sum(axis=0).A1
            
            # Element-wise subtraction; assumes feature_names aligns for both sums
            adjusted_feature_scores -= contrast_feature_scores_sum

    # Map scores to feature names
    feature_score_dict = dict(zip(feature_names, adjusted_feature_scores))
    # Sort features by score
    sorted_features = sorted(feature_score_dict.items(), key=lambda item: item[1], reverse=True)

    # Return the names of the top_n features that have a score > 0
    top_features = [feature for feature, score in sorted_features if score > 0][:top_n]

    return top_features

def compute_clusters_style_representation_2(
        background_corpus_df: pd.DataFrame,
        cluster_ids: List[Any],
        cluster_label_clm_name: str = 'cluster_label',
        max_num_feats: int = 5,
        max_num_documents_per_author=3,
        max_num_authors=5):
    """
    Call openAI to analyze the common writing style features of the given list of texts
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    assert background_corpus_df['fullText'].apply(
        lambda x: isinstance(x, list) and all(isinstance(feat, str) for feat in x)
    ).all(), f"Column fullText must contain lists of strings."

    background_corpus_df = background_corpus_df[background_corpus_df[cluster_label_clm_name].isin(cluster_ids)]
    
    author_texts = ['\n\n'.join(author[:max_num_documents_per_author]) for author in background_corpus_df['fullText'].tolist()[:max_num_authors]]
    author_texts = "\n\n".join(["""Author {}:\n""".format(i+1) + text for i, text in enumerate(author_texts)])
    author_names = background_corpus_df[cluster_label_clm_name].tolist()[:max_num_authors]
    print(author_names)
    
    prompt = f"""First identify a list of {max_num_feats} writing style features that are common between the given texts. Second for every text and style feature, extract all spans that represent the feature.
    Author Texts:
    \"\"\"{author_texts}\"\"\"
    """

    # Compute MD5 hash
    digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{digest}.pkl")

    # If cache hit, load and return
    if os.path.exists(cache_path):
        print(f"Loading authors writing style from cache ...")
        with open(cache_path, "rb") as f:
            parsed_response = pickle.load(f)
    
    else: # Else compute and cache

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"assistant","content":"You are a forensic linguistic who knows how to analyze similarites in writing styles."},
                {"role":"user","content":prompt}],
            response_format={"type": "json_schema", "json_schema": {"name": "style_analysis_schema", "schema":  to_strict_json_schema(style_analysis_schema)}}
        )

        parsed_response = json.loads(response.choices[0].message.content)

        with open(cache_path, "wb") as f:
            pickle.dump(parsed_response, f)

    return parsed_response 


def compute_clusters_g2v_representation(
    background_corpus_df: pd.DataFrame,
    author_ids: List[Any],
    other_author_ids: List[Any],
    features_clm_name: str,
    top_n: int = 10
) -> List[str]:


    # Get boolean mask for documents in selected clusters
    selected_mask = background_corpus_df['authorID'].isin(author_ids).to_numpy()

    if not selected_mask.any():
        return [] # No documents found for the given cluster_ids

    selected_feats = background_corpus_df[selected_mask][features_clm_name].tolist()
    all_g2v_feats  = list(selected_feats[0].keys())
    all_g2v_values = np.array([list(x.values()) for x in selected_feats]).mean(axis=0)


    other_selected_feats = background_corpus_df[~selected_mask][features_clm_name].tolist()
    all_g2v_other_feats  = list(other_selected_feats[0].keys())
    all_g2v_other_values = np.array([list(x.values()) for x in other_selected_feats]).mean(axis=0)

    final_g2v_feats_values = all_g2v_values - all_g2v_other_values


    top_g2v_feats = sorted(list(zip(all_g2v_feats, final_g2v_feats_values)), key=lambda x: -x[1])
    print(top_g2v_feats[:top_n])

    return [x[0] for x in top_g2v_feats[:top_n]]


def generate_interpretable_space_representation(interp_space_path, styles_df_path, feat_clm, output_clm, num_feats=5):
    
    styles_df = pd.read_csv(styles_df_path)[[feat_clm, "documentID"]]

    # A dictionary of style features and their IDF
    style_feats_agg_df = styles_df.groupby(feat_clm).agg({'documentID': lambda x : len(list(x))}).reset_index()
    style_feats_agg_df['document_freq'] = style_feats_agg_df.documentID
    style_to_feats_dfreq = {x[0]: math.log(styles_df.documentID.nunique()/x[1]) for x in zip(style_feats_agg_df[feat_clm].tolist(), style_feats_agg_df.document_freq.tolist())}
    
    # A list of style features we work with
    style_feats_list = style_feats_agg_df[feat_clm].tolist()
    print('Number of style feats ', len(style_feats_list))
    
    # A list of documents and what list of style features each has
    doc_style_agg_df     = styles_df.groupby('documentID').agg({feat_clm: lambda x : list(x)}).reset_index()
    document_to_feats_dict = {x[0]: x[1] for x in zip(doc_style_agg_df.documentID.tolist(), doc_style_agg_df[feat_clm].tolist())}
    
    

    # Load the clustering information
    df = pd.read_pickle(interp_space_path)
    df = df[df.cluster_label != -1]
    # A cluster to list of documents
    clusterd_df = df.groupby('cluster_label').agg({
        'documentID': lambda x: [d_id for doc_ids in x for d_id in doc_ids]
    }).reset_index()
    
    # Filter-in only documents that has a style description
    clusterd_df['documentID'] = clusterd_df.documentID.apply(lambda documentIDs: [documentID for documentID in documentIDs if documentID in document_to_feats_dict])
    # Map from cluster label to list of features through the document information
    clusterd_df[feat_clm] = clusterd_df.documentID.apply(lambda doc_ids: [f for d_id in doc_ids for f in document_to_feats_dict[d_id]])

    def compute_tfidf(row):
        style_counts = Counter(row[feat_clm])
        total_num_styles = sum(style_counts.values())
        #print(style_counts, total_num_styles)
        style_distribution = {
            style: math.log(1+count) * style_to_feats_dfreq[style] if style in style_to_feats_dfreq else 0 for style, count in style_counts.items()
        } #TF-IDF
        
        return style_distribution

    def create_tfidf_rep(tfidf_dist, num_feats):
        style_feats = sorted(tfidf_dist.items(), key=lambda x: -x[1])
        top_k_feats = [x[0] for x in style_feats[:num_feats] if str(x[0]) != 'nan']
        return top_k_feats

    clusterd_df[output_clm +'_dist'] = clusterd_df.apply(lambda row: compute_tfidf(row), axis=1)
    clusterd_df[output_clm]         = clusterd_df[output_clm +'_dist'].apply(lambda dist: create_tfidf_rep(dist, num_feats))

        
    return clusterd_df

if __name__ == "__main__":
    background_corpus = pd.read_pickle('../datasets/luar_interp_space_cluster_19/train_authors.pkl')
    print(background_corpus.columns)
    print(background_corpus[['authorID', 'fullText', 'cluster_label']].head())
    # # Example: Find features for clusters [2,3,4] that are NOT prominent in cluster [1]
    # feats = compute_clusters_style_representation(
    #     background_corpus_df=background_corpus,
    #     cluster_ids=['00005a5c-5c06-3a36-37f9-53c6422a31d8',],
    #     other_cluster_ids=[], # Pass the contrastive cluster IDs here
    #     cluster_label_clm_name='authorID',
    #     features_clm_name='final_attribute_name'
    # )
    # print(feats)
    generate_style_embedding(background_corpus, 'fullText', 'AnnaWegmann/Style-Embedding')
    print(background_corpus.columns)
