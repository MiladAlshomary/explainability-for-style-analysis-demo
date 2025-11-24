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
from pydantic import ValidationError
import time
from utils.llm_feat_utils import generate_feature_spans_cached
from utils.gram2vec_feat_utils import get_shorthand, get_fullform
from gram2vec.feature_locator import find_feature_spans
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA

CACHE_DIR = "datasets/embeddings_cache"
G2V_CACHE = "datasets/gram2vec_cache"
ZOOM_CACHE = "datasets/zoom_cache/features_cache.json"
REGION_CACHE = "datasets/region_cache/regions_cache.pkl"
SUMMARY_CACHE = "datasets/summary_cache/summaries.json"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(G2V_CACHE, exist_ok=True)
os.makedirs(os.path.dirname(ZOOM_CACHE), exist_ok=True)
os.makedirs(os.path.dirname(REGION_CACHE), exist_ok=True)
# Bump this whenever there is a change etc...
CACHE_VERSION = 1

class style_analysis_schema(BaseModel):
    features: list[str]
    spans: dict[str, dict[str, list[str]]]

class FeatureIdentificationSchema(BaseModel):
    features: list[str]

class SpanExtractionSchema(BaseModel):
    spans: dict[str, dict[str, list[str]]]  # {author_name: {feature: [spans]}}

class StyleSummarySchema(BaseModel):
    summary_paragraph: str



def compute_g2v_features(clustered_authors_df: pd.DataFrame, task_authors_df: pd.DataFrame=None, text_clm='fullText') -> pd.DataFrame:
    """
    Computes gram2vec feature vectors for each author and adds them to the DataFrame.
    This effectively creates a mapping from each author to their vector.
    """
    if task_authors_df is not None:
        print (f"concatenating task authors and background corpus authors")
        print(f"Number of task authors: {len(task_authors_df)}")
        print(f"task authors author_ids: {task_authors_df.authorID.tolist()}")
        # print(f"task authors -->")
        # print(task_authors_df)
        print(f"Number of background corpus authors: {len(clustered_authors_df)}")
        clustered_authors_df = pd.concat([task_authors_df, clustered_authors_df])
        print(f"Number of authors after concatenation: {len(clustered_authors_df)}")

    # Gather the input texts (preserves list-of-strings if any)
    #texts = background_corpus_df[text_clm].fillna("").tolist()
    author_texts = ['\n\n'.join(x) for x in clustered_authors_df.fullText.tolist()]
    # print('author_text at 0:{}'.format(author_texts[0]))
    print(f"Number of author_texts: {len(author_texts)}")

    # Create a reproducible JSON serialization of the texts
    # why are g2v features going into a new file inside embeddings_cache?
    # changed to G2V_CACHE
    serialized = json.dumps({
        "col": text_clm,
        "texts": author_texts
    }, sort_keys=True, ensure_ascii=False)

    # Compute MD5 hash
    digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()
    cache_path = os.path.join(G2V_CACHE, f"{digest}.pkl")

    # If cache hit, load and return
    if os.path.exists(cache_path):
        # print(f"Cache hit...")
        # Making this green to make it stand out from rest of the logs
        print(f"\n\n\n\033[1m\033[92m>>> Cache hit for {cache_path} <<<\033[0m\n")
        with open(cache_path, "rb") as f:
            clustered_authors_df = pickle.load(f)
    
    else: # Else compute and cache
        # Making this red to make it stand out from rest of the logs
        print(f"\n\n\n\033[1m\033[91m>>> Cache miss for {cache_path} => Computing fresh!! <<<\033[0m\n")

        g2v_feats_df = vectorizer.from_documents(author_texts, batch_size=8)

        print(f"Number of g2v features: {len(g2v_feats_df)}")
        print(f"Number of clustered_authors_df.authorID.tolist(): {len(clustered_authors_df.authorID.tolist())}")
        print(f"Number of g2v_feats_df.to_numpy().tolist(): {len(g2v_feats_df.to_numpy().tolist())}")

        ids = clustered_authors_df.authorID.tolist()
        counter = Counter(ids)
        duplicates = [k for k, v in counter.items() if v > 1]

        print(f"Duplicate authorIDs: {duplicates}")
        print(f"Number of duplicates: {len(ids) - len(set(ids))}")

        author_to_g2v_feats = {x[0]: x[1] for x in zip(clustered_authors_df.authorID.tolist(), g2v_feats_df.to_numpy().tolist())}

        print(f"Number of authors with g2v features: {len(author_to_g2v_feats)}")

        # apply normalization
        vector_std  = np.std(list(author_to_g2v_feats.values()), axis=0)
        vector_mean = np.mean(list(author_to_g2v_feats.values()), axis=0)
        vector_std[vector_std == 0] = 1.0
        author_to_g2v_feats_z_normalized = {x[0]: (x[1] - vector_mean) / vector_std for x in author_to_g2v_feats.items()}

        print(f"Number of authors with g2v features normalized: {len(author_to_g2v_feats_z_normalized)}")
        print(f" len of clustered authors df: {len(clustered_authors_df)}")
        

        # Add the vectors as a new column of the DataFrame.
        clustered_authors_df['g2v_vector'] = [{x[1]: x[0] for x in zip(val, g2v_feats_df.columns.tolist())} 
                                            for val in author_to_g2v_feats_z_normalized.values()]
        
        with open(cache_path, "wb") as f:
            pickle.dump(clustered_authors_df, f)
            # Making this green to make it stand out from rest of the logs
            print(f"\n\n\n\033[1m\033[92m>>> Saved to {cache_path} <<<\033[0m\n")
            # the file generated here contains g2v + style embeddings. 
    
    if task_authors_df is not None:
        task_authors_df = clustered_authors_df[clustered_authors_df.authorID.isin(task_authors_df.authorID.tolist())]
        clustered_authors_df = clustered_authors_df[~clustered_authors_df.authorID.isin(task_authors_df.authorID.tolist())]


    return clustered_authors_df['g2v_vector'].tolist(), task_authors_df['g2v_vector'].tolist()


def get_task_authors_from_background_df(background_df):
    task_authors_df = background_df[background_df.authorID.isin(["Q_author", "a0_author", "a1_author", "a2_author"])]
    return task_authors_df

def instance_to_df(instance, predicted_author=None, ground_truth_author=None):
    #create a dataframe of the task authors
    task_authos_df  = pd.DataFrame([
        {'authorID': 'Mystery author', 'fullText': instance['Q_fullText'], 'predicted': None, 'ground_truth': None},
        {'authorID': 'Candidate Author 1', 'fullText': instance['a0_fullText'], 'predicted': int(predicted_author) == 0 if predicted_author is not None else None, 'ground_truth': int(ground_truth_author) == 0 if ground_truth_author is not None else None},
        {'authorID': 'Candidate Author 2', 'fullText': instance['a1_fullText'], 'predicted': int(predicted_author) == 1 if predicted_author is not None else None, 'ground_truth': int(ground_truth_author) == 1 if ground_truth_author is not None else None},
        {'authorID': 'Candidate Author 3', 'fullText': instance['a2_fullText'], 'predicted': int(predicted_author) == 2 if predicted_author is not None else None, 'ground_truth': int(ground_truth_author) == 2 if ground_truth_author is not None else None}
                    
    ])

    # if type(instance['Q_fullText']) == list:
    #     task_authos_df = task_authos_df.groupby('authorID').agg({'fullText': lambda x: list(x)}).reset_index()

    return task_authos_df


def generate_style_embedding(background_corpus_df: pd.DataFrame, text_clm: str, model_name: str, dimensionality_reduction: bool = True, dimensions: int = 100) -> pd.DataFrame:
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

    # Apply PCA over the embeddings to reduce the dimentionality 
    if dimensionality_reduction:
        if len(final_embeddings) > 0 and len(final_embeddings[0]) > dimensions: # Only apply PCA if embeddings exist and dim > dimensions
            pca = PCA(n_components=dimensions)
            final_embeddings = pca.fit_transform(final_embeddings)

    return list(final_embeddings)

# ── wrapper with caching ───────────────────────────────────────
def cached_generate_style_embedding(background_corpus_df: pd.DataFrame,
                                    text_clm: str,
                                    model_name: str,
                                    task_authors_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Wraps `generate_style_embedding`, caching its output in pickle files
    keyed by an MD5 of (model_name + text list). If the cache exists,
    loads and returns it instead of recomputing.
    """

    if task_authors_df is not None:
        print (f"concatenating task authors and background corpus authors")
        print(f"Number of task authors: {len(task_authors_df)}")
        print(f"task authors author_ids: {task_authors_df.authorID.tolist()}")
        print(f"Number of background corpus authors: {len(background_corpus_df)}")
        background_corpus_df = pd.concat([task_authors_df, background_corpus_df])
        print(f"Number of authors after concatenation: {len(background_corpus_df)}")

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
        # Making this green to make it stand out from rest of the logs
        print(f"\n\n\n\033[1m\033[92m>>> Cache hit for {cache_path} for {model_name} on column '{text_clm} <<<\033[0m\n")
        with open(cache_path, "rb") as f:
            background_corpus_df = pickle.load(f)

    else:
        # Otherwise, compute, cache, and return
        print(f"\n\n\n\033[1m\033[91m>>> Cache miss for {cache_path} for {model_name} on column '{text_clm} <<<\033[0m\n")
        task_and_background_embeddings = generate_style_embedding(background_corpus_df, text_clm, model_name, dimensionality_reduction=False)
        # Create a clean column name from the model name
        col_name = f'{model_name.split("/")[-1]}_style_embedding'
        background_corpus_df[col_name] = task_and_background_embeddings

        with open(cache_path, "wb") as f:
            pickle.dump(background_corpus_df, f)
            print(f"\n\n\n\033[1m\033[92m>>> Cache saved for {cache_path} for {model_name} on column '{text_clm} <<<\033[0m\n")
    
    if task_authors_df is not None:
        task_authors_df = background_corpus_df[background_corpus_df.authorID.isin(task_authors_df.authorID.tolist())]
        background_corpus_df = background_corpus_df[~background_corpus_df.authorID.isin(task_authors_df.authorID.tolist())]

    return background_corpus_df, task_authors_df

# Noticed the following function isnt actually referenced anywhere.
# def get_style_feats_distribution(documentIDs, style_feats_dict):
#     style_feats = []
#     for documentId in documentIDs:
#         if documentId not in document_to_style_feats:
#             #print(documentId)
#             continue

#         style_feats+= document_to_style_feats[documentId]

#     tfidf = [style_feats.count(key) * val for key, val in style_feats_dict.items()]

#     return tfidf
# 
# Noticed the following function isnt actually referenced anywhere.
# def get_cluster_top_feats(style_feats_distribution, style_feats_list, top_k=5):
#     sorted_feats = np.argsort(style_feats_distribution)[::-1]
#     top_feats = [style_feats_list[x] for x in sorted_feats[:top_k] if style_feats_distribution[x] > 0]
#     return top_feats

# Noticed the following function isnt actually referenced anywhere.
# def compute_clusters_style_representation(
#     background_corpus_df: pd.DataFrame,
#     cluster_ids: List[Any],
#     other_cluster_ids: List[Any],
#     features_clm_name: str,
#     cluster_label_clm_name: str = 'cluster_label',
#     top_n: int = 10
# ) -> List[str]:
#     """
#     Given a DataFrame with document IDs, cluster IDs, and feature lists,
#     return the top N features that are most important in the specified `cluster_ids`
#     while having low importance in `other_cluster_ids`.
#     Importance is determined by TF-IDF scores. The final score for a feature is
#     (summed TF-IDF in `cluster_ids`) - (summed TF-IDF in `other_cluster_ids`).

#     Parameters:
#     - background_corpus_df: pd.DataFrame. Must contain the columns specified by
#                             `cluster_label_clm_name` and `features_clm_name`.
#                             The column `features_clm_name` should contain lists of strings (features).
#     - cluster_ids: List of cluster IDs for which to find representative features (target clusters).
#     - other_cluster_ids: List of cluster IDs whose features should be down-weighted.
#                          Features prominent in these clusters will have their scores reduced.
#                          Pass an empty list or None if no contrastive clusters are needed.
#     - features_clm_name: The name of the column in `background_corpus_df` that
#                          contains the list of features for each document.
#     - cluster_label_clm_name: The name of the column in `background_corpus_df`
#                               that contains the cluster labels. Defaults to 'cluster_label'.
#     - top_n: Number of top features to return.
#     Returns:
#     - List[str]: A list of feature names. These are up to `top_n` features
#                  ranked by their adjusted TF-IDF scores (score in `cluster_ids`
#                  minus score in `other_cluster_ids`). Only features with a final
#                  adjusted score > 0 are included.
#     """

#     assert background_corpus_df[features_clm_name].apply(
#         lambda x: isinstance(x, list) and all(isinstance(feat, str) for feat in x)
#     ).all(), f"Column '{features_clm_name}' must contain lists of strings."

#     # Compute TF-IDF on the entire corpus
#     vectorizer = TfidfVectorizer(
#         tokenizer=lambda x: x,
#         preprocessor=lambda x: x,
#         token_pattern=None  # Disable default token pattern, treat items in list as tokens
#     )
#     tfidf_matrix = vectorizer.fit_transform(background_corpus_df[features_clm_name])
#     feature_names = vectorizer.get_feature_names_out()

#     # Get boolean mask for documents in selected clusters
#     selected_mask = background_corpus_df[cluster_label_clm_name].isin(cluster_ids).to_numpy()

#     if not selected_mask.any():
#         return [] # No documents found for the given cluster_ids

#     # Subset the TF-IDF matrix using the boolean mask
#     selected_tfidf = tfidf_matrix[selected_mask]

#     # Sum TF-IDF scores across documents for each feature in the target clusters
#     target_feature_scores_sum = selected_tfidf.sum(axis=0).A1  # Convert to 1D array

#     # Initialize adjusted scores with target scores
#     adjusted_feature_scores = target_feature_scores_sum.copy()

#     # If other_cluster_ids are provided and not empty, subtract their TF-IDF sums
#     if other_cluster_ids: # Checks if the list is not None and not empty
#         other_selected_mask = background_corpus_df[cluster_label_clm_name].isin(other_cluster_ids).to_numpy()

#         if other_selected_mask.any():
#             other_selected_tfidf = tfidf_matrix[other_selected_mask]
#             contrast_feature_scores_sum = other_selected_tfidf.sum(axis=0).A1
            
#             # Element-wise subtraction; assumes feature_names aligns for both sums
#             adjusted_feature_scores -= contrast_feature_scores_sum

#     # Map scores to feature names
#     feature_score_dict = dict(zip(feature_names, adjusted_feature_scores))
#     # Sort features by score
#     sorted_features = sorted(feature_score_dict.items(), key=lambda item: item[1], reverse=True)

#     # Return the names of the top_n features that have a score > 0
#     top_features = [feature for feature, score in sorted_features if score > 0][:top_n]

#     return top_features

# Noticed the following function isnt actually referenced anywhere.
# def compute_clusters_style_representation_2(
#         background_corpus_df: pd.DataFrame,
#         cluster_ids: List[Any],
#         cluster_label_clm_name: str = 'cluster_label',
#         max_num_feats: int = 5,
#         max_num_documents_per_author=3,
#         max_num_authors=5):
#     """
#     Call openAI to analyze the common writing style features of the given list of texts
#     """
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     background_corpus_df['fullText'] = background_corpus_df['fullText'].map(lambda x: '\n\n'.join(x[:max_num_documents_per_author]) if isinstance(x, list) else x)
#     background_corpus_df = background_corpus_df[background_corpus_df[cluster_label_clm_name].isin(cluster_ids)]
    
#     author_texts = background_corpus_df['fullText'].tolist()[:max_num_authors]
#     author_texts = "\n\n".join(["""Author {}:\n""".format(i+1) + text for i, text in enumerate(author_texts)])
#     author_names = background_corpus_df[cluster_label_clm_name].tolist()[:max_num_authors]
#     print(f"Number of authors: {len(background_corpus_df)}")
#     print(author_names)
#     print(author_texts)
#     print(f"Number of authors: {len(author_names)}")
#     print(f"Number of authors: {len(author_texts)}")
    
#     prompt = f"""First identify a list of {max_num_feats} writing style features that are common between the given texts. Second for every author text and style feature, extract all spans that represent the feature. Output for every author all style features with their spans.
#     Author Texts:
#     \"\"\"{author_texts}\"\"\"
#     """

#     # Compute MD5 hash
#     digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
#     cache_path = os.path.join(CACHE_DIR, f"{digest}.pkl")

#     # If cache hit, load and return
#     if os.path.exists(cache_path):
#         print(f"Loading authors writing style from cache ...")
#         with open(cache_path, "rb") as f:
#             parsed_response = pickle.load(f)
    
#     else: # Else compute and cache

#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role":"assistant","content":"You are a forensic linguistic who knows how to analyze similarites in writing styles."},
#                 {"role":"user","content":prompt}],
#             response_format={"type": "json_schema", "json_schema": {"name": "style_analysis_schema", "schema":  to_strict_json_schema(style_analysis_schema)}}
#         )

#         parsed_response = json.loads(response.choices[0].message.content)

#         with open(cache_path, "wb") as f:
#             pickle.dump(parsed_response, f)

#     return parsed_response 

def generate_cache_key(author_names: List[str], max_num_feats: int) -> str:
    """Generate a unique cache key based on author names and max features"""
    # Sort author names to ensure consistent key regardless of order
    sorted_authors = sorted(author_names)
    key_data = {
        "authors": sorted_authors,
        "max_num_feats": max_num_feats
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()

def identify_style_features(author_texts: list[str], author_names: list[str], max_num_feats: int = 5) -> list[str]:
    cache_key = None
    if author_names:
        cache_key = generate_cache_key(author_names, max_num_feats)

        if os.path.exists(ZOOM_CACHE):
            with open(ZOOM_CACHE, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

        if cache_key in cache:
            print(f"\nCache hit! Using cached features for authors: {author_names}")
            print(f"\n\n\n\033[1m\033[92m>>> Cache hit for {cache_key} in {ZOOM_CACHE} <<<\033[0m\n")
            return cache[cache_key]["features"]
        else:
            print(f"\n\n\n\033[1m\033[91m>>> Cache miss for {cache_key} in {ZOOM_CACHE} \nComputing features for authors: {author_names}<<<\033[0m\n")

    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", None), api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""Identify {max_num_feats} writing style features that are common between the authors texts.
    Author Texts:

    {author_texts}
    """

    # print('==================>>>>>>>>>>')
    # print(prompt)
    # print('==================>>>>>>>>>>')
    def _make_call():
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "assistant", "content": "You are a forensic linguist who knows how to analyze linguistic and stylometric similarites between texts."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "FeatureIdentificationSchema",
                    "schema": to_strict_json_schema(FeatureIdentificationSchema)
                }
            }
        )
        return json.loads(response.choices[0].message.content)

    features = retry_call(_make_call, FeatureIdentificationSchema).features

    if cache_key and author_names:
        cache[cache_key] = {
            "features": features
        }
        # save_cache(cache)
        with open(ZOOM_CACHE, 'w') as f:
            json.dump(cache, f, indent=2)
            print(f"\n\n\n\033[1m\033[92m>>> Cache saved for {cache_key} in {ZOOM_CACHE}<<<\033[0m\n")
            

        print(f"Cached features for authors: {author_names}")
    
    return features

def retry_call(call_fn, schema_class, max_attempts=3, wait_sec=2):
    for attempt in range(max_attempts):
        try:
            result = call_fn()
            # Validate against schema
            validated = schema_class(**result)
            return validated
        except (ValidationError, KeyError, json.JSONDecodeError) as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(wait_sec)
    raise RuntimeError("All retry attempts failed for OpenAI call.")

def extract_all_spans(authors_df: pd.DataFrame, features: list[str], cluster_label_clm_name: str = 'authorID') -> dict[str, dict[str, list[str]]]:
    """
    For each author, use `generate_feature_spans_cached` to get feature->span mappings.
    Returns a dict: {author_name: {feature: [spans]}}
    """
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", None), api_key=os.getenv("OPENAI_API_KEY"))

    spans_by_author = {}

    for _, row in authors_df.iterrows():
        author_name = str(row[cluster_label_clm_name])
        # print(author_name)
        role = f"{author_name}"
        full_text = row['fullText']
        spans = generate_feature_spans_cached(client, full_text, features, role)
        spans_by_author[author_name] = spans

    return spans_by_author

def compute_clusters_style_representation_3(
    background_corpus_df: pd.DataFrame,
    cluster_ids: List[Any],
    cluster_label_clm_name: str = 'authorID',
    max_num_feats: int = 25,
    max_num_documents_per_author=10,
    max_num_authors=10,
    max_authors_for_span_extraction=4,
    top_k: int = 10,
    predicted_author = None,
    return_only_feats= False
    ):

    print(f"Computing style representation for visible clusters: {len(cluster_ids)}")
    # STEP 1: Identify features on max_num_authors's max_num_documents_per_author number of documents
    background_corpus_df['fullText'] = background_corpus_df['fullText'].map(lambda x: '\n\n'.join(x[:max_num_documents_per_author]) if isinstance(x, list) else x)
    background_corpus_df_feat_id = background_corpus_df[background_corpus_df[cluster_label_clm_name].isin(cluster_ids)]
    
    author_texts = background_corpus_df_feat_id['fullText'].tolist()[:max_num_authors]
    author_texts = "\n\n".join(["""Author {}:\n""".format(i+1) + text for i, text in enumerate(author_texts)])
    author_names = background_corpus_df_feat_id[cluster_label_clm_name].tolist()[:max_num_authors]
    print(f"Number of authors: {len(background_corpus_df_feat_id)}")
    # print(author_names)
    features = identify_style_features(author_texts, author_names, max_num_feats=max_num_feats)

    if return_only_feats:
        return features
    
    #print("Features: ", features)
    # STEP 2: Prepare author pool for span extraction
    span_df = background_corpus_df.iloc[:max_authors_for_span_extraction]
    author_names = span_df[cluster_label_clm_name].tolist()[:max_authors_for_span_extraction]
    print(f"Number of authors for span detection : {len(span_df)}")
    # print(author_names)
    spans_by_author = extract_all_spans(span_df, features, cluster_label_clm_name)
    
    # Filter-in only task authors that are part of the current selection
    task_author_names = {'Mystery author', 'Candidate Author 1', 'Candidate Author 2', 'Candidate Author 3'}
    
    
    # Compute feature importance based on Mystery + Predicted author vs. other candidates
    feature_importance = {f : 0 for f in features}
    for author, feature_map in spans_by_author.items():
        if author in task_author_names.intersection(set(cluster_ids)):
            for feature, spans in feature_map.items():
                if spans:
                    feature_importance[feature] += len(spans)
        else:
            # Background authors - subtract their span counts
            for feature, spans in feature_map.items():
                if spans:
                    feature_importance[feature] -= len(spans)
    
    print(f"Feature importance scores: {feature_importance}")
    selected_features_ranked = sorted(feature_importance, key=lambda f: -feature_importance[f])[:int(top_k)]

    #print('filtered set of features (min coverage', len(author_present_feature_sets), '): ', selected_features_ranked)

    return {
        "features": list(selected_features_ranked),
        "spans": spans_by_author
    }

def summarize_style_features_to_paragraph(features: list[str]) -> str:
    """
    Takes a list of writing style features and uses an LLM to generate a
    coherent, descriptive paragraph summarizing the style.

    Args:
        features (list[str]): A list of style features.

    Returns:
        str: A single paragraph summarizing the writing style.
    """
    if not features:
        return "No style features were identified for this selection."

    # Generate a cache key based on the sorted features to ensure consistency
    feature_key = hashlib.md5(json.dumps(sorted(features)).encode()).hexdigest()
    
    os.makedirs(os.path.dirname(SUMMARY_CACHE), exist_ok=True)
    if os.path.exists(SUMMARY_CACHE):
        with open(SUMMARY_CACHE, 'r') as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {}
    else:
        cache = {}

    if feature_key in cache:
        print(f"Cache hit for style summary. Key: {feature_key}")
        return cache[feature_key]

    print(f"Cache miss for style summary. Generating new summary...")
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", None), api_key=os.getenv("OPENAI_API_KEY"))

    feature_list_str = "\n".join([f"- {feat}" for feat in features])
    prompt = f"""You are a linguistic analyst. Your task is to synthesize the following list of writing style features into a single, coherent, and descriptive paragraph. The paragraph should flow naturally and explain the overall writing style of an author based on these features. Be concise and only mention the features without referring to example spans.

Style Features:
{feature_list_str}

Please provide the summary as a single paragraph.
"""

    def _make_call():
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {"name": "StyleSummarySchema", "schema": to_strict_json_schema(StyleSummarySchema)}}
        )
        return json.loads(response.choices[0].message.content)

    summary_paragraph = retry_call(_make_call, StyleSummarySchema).summary_paragraph
    
    # Save to cache
    cache[feature_key] = summary_paragraph
    with open(SUMMARY_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)

    return summary_paragraph

def find_closest_cluster_style(texts: list[str], interp_space, model_name: str) -> str:
    """
    Computes the average embedding for a list of texts and finds the most similar
    cluster from the interpretable space, returning its style description.

    Args:
        texts (list[str]): A list of texts for which to find a style description.
        interp_space_path (str): Path to the interpretable_space.json file.
        model_name (str): The name of the sentence transformer model to use for embeddings.

    Returns:
        str: The style description paragraph of the most similar cluster.
    """
    if not texts:
        return "No texts provided for analysis."


    # 2. Compute the average embedding for the input texts
    # We create a temporary DataFrame to use the existing embedding generation utility
    temp_df = pd.DataFrame([{'fullText': texts}])
    input_embedding_list = generate_style_embedding(temp_df, 'fullText', model_name, dimensionality_reduction=False)
    
    if not input_embedding_list:
        return "Could not generate an embedding for the provided texts."
        
    input_embedding = np.array(input_embedding_list[0]).reshape(1, -1)

    # 3. Find the most similar cluster
    cluster_embeddings = {int(k): np.array(v[0]) for k, v in interp_space.items()}
    
    best_cluster_label = -1
    max_similarity = -1

    for label, cluster_emb in cluster_embeddings.items():
        similarity = cosine_similarity(input_embedding, cluster_emb.reshape(1, -1))[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_cluster_label = label

    # 4. Return the style description of the closest cluster
    return interp_space.get(str(best_cluster_label), [None, "Could not find a matching style description."])[1], input_embedding[0]


def compute_clusters_g2v_representation(
    background_corpus_df: pd.DataFrame,
    author_ids: List[Any],
    other_author_ids: List[Any],
    features_clm_name: str,
    top_n: int = 15,
    max_candidates_for_span_sorting: int = 50,
    predicted_author: int = None
) -> List[tuple]:  # Changed return type to List[tuple] to include scores

    print(f"[INFO] Computing G2V representation with predicted_author: {predicted_author}")
    # 1) Identify selected authors in the zoom region
    selected_mask = background_corpus_df['authorID'].isin(author_ids).to_numpy()

    if not selected_mask.any():
        return []  # No authors found for the given author_ids

    # 2) Build a population matrix of all authors' Gram2Vec features
    #    Expect each row in features_clm_name to be a dict {feature_name: value}
    all_feature_dicts = background_corpus_df[features_clm_name].tolist()
    if not all_feature_dicts:
        return []

    #    Use the first row to get consistent feature ordering
    all_features = list(all_feature_dicts[0].keys())
    population_matrix = np.array(
        [[feat_dict.get(feat, 0.0) for feat in all_features] for feat_dict in all_feature_dicts],
        dtype=float
    )

    # 3) Z-normalize columnwise across the entire corpus
    col_means = population_matrix.mean(axis=0)
    col_stds = population_matrix.std(axis=0)
    col_stds[col_stds == 0] = 1.0
    z_population = (population_matrix - col_means) / col_stds

    # 4) Take the mean across the selected authors (zoom region)
    selected_mean = z_population[selected_mask].mean(axis=0)

    # 5) Rank features by mean z-score, keep positives only
    feature_scores = [(feat, float(score)) for feat, score in zip(all_features, selected_mean) if score > 0]
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 6) Extract top candidates for span-based sorting
    candidate_features = feature_scores[:top_n]
    
    # 7) Extract spans for task authors to sort by frequency
    task_author_names = {'Mystery author', 'Candidate Author 1', 'Candidate Author 2', 'Candidate Author 3'}
    task_authors_in_selection = task_author_names.intersection(set(author_ids))
    
    if not task_authors_in_selection:
        # If no task authors in selection, just return the z-score sorted features
        print("[INFO] No task authors in selection, returning z-score sorted features")
        return feature_scores[:top_n]
    
    # Get task author data
    task_authors_df = background_corpus_df[background_corpus_df['authorID'].isin(task_author_names)]
    
    
    print('len of task_authors_df ', len(task_authors_df))
    print('zoomed in authors {}'.format(task_authors_in_selection))
    # Count spans for each feature: +1 for Mystery/Predicted, -1 for other candidates
    feature_span_scores = {}
    for feat_shorthand, _ in candidate_features:
        span_score = 0
        
        for _, author_row in task_authors_df.iterrows():
            author_name = author_row['authorID']
            author_text = author_row['fullText']
            if isinstance(author_text, list):
                author_text = '\n\n'.join(author_text)
            
            try:
                # find_feature_spans expects shorthand format like "pos_unigrams:ADJ"
                spans = find_feature_spans(author_text, feat_shorthand)
                span_count = len(spans)
                # Add span count if Mystery or Predicted author, subtract if other candidate
                if author_name in task_authors_in_selection:
                    span_score += span_count
                else:
                    # Other candidates - subtract their span counts
                    span_score -= span_count
            except Exception as e:
                # If span extraction fails, continue with 0 spans for this author
                pass
        
        feature_span_scores[feat_shorthand] = span_score
    
    # 8) Sort features by span score (Mystery+Predicted vs Others), then by z-score as tiebreaker
    sorted_by_spans = sorted(
        candidate_features,
        key=lambda x: (-feature_span_scores.get(x[0], 0), -x[1])
    )
    
    print(f"[INFO] Top 5 gram2vec features by span score: {[(f, feature_span_scores.get(f, 0), z) for f, z in sorted_by_spans[:5]]}")
    
    return sorted_by_spans

# Noticed the following function isnt actually referenced anywhere.
# def generate_interpretable_space_representation(interp_space_path, styles_df_path, feat_clm, output_clm, num_feats=5):
    
#     styles_df = pd.read_csv(styles_df_path)[[feat_clm, "documentID"]]

#     # A dictionary of style features and their IDF
#     style_feats_agg_df = styles_df.groupby(feat_clm).agg({'documentID': lambda x : len(list(x))}).reset_index()
#     style_feats_agg_df['document_freq'] = style_feats_agg_df.documentID
#     style_to_feats_dfreq = {x[0]: math.log(styles_df.documentID.nunique()/x[1]) for x in zip(style_feats_agg_df[feat_clm].tolist(), style_feats_agg_df.document_freq.tolist())}
    
#     # A list of style features we work with
#     style_feats_list = style_feats_agg_df[feat_clm].tolist()
#     print('Number of style feats ', len(style_feats_list))
    
#     # A list of documents and what list of style features each has
#     doc_style_agg_df     = styles_df.groupby('documentID').agg({feat_clm: lambda x : list(x)}).reset_index()
#     document_to_feats_dict = {x[0]: x[1] for x in zip(doc_style_agg_df.documentID.tolist(), doc_style_agg_df[feat_clm].tolist())}
    
    

#     # Load the clustering information
#     df = pd.read_pickle(interp_space_path)
#     df = df[df.cluster_label != -1]
#     # A cluster to list of documents
#     clusterd_df = df.groupby('cluster_label').agg({
#         'documentID': lambda x: [d_id for doc_ids in x for d_id in doc_ids]
#     }).reset_index()
    
#     # Filter-in only documents that has a style description
#     clusterd_df['documentID'] = clusterd_df.documentID.apply(lambda documentIDs: [documentID for documentID in documentIDs if documentID in document_to_feats_dict])
#     # Map from cluster label to list of features through the document information
#     clusterd_df[feat_clm] = clusterd_df.documentID.apply(lambda doc_ids: [f for d_id in doc_ids for f in document_to_feats_dict[d_id]])

#     def compute_tfidf(row):
#         style_counts = Counter(row[feat_clm])
#         total_num_styles = sum(style_counts.values())
#         #print(style_counts, total_num_styles)
#         style_distribution = {
#             style: math.log(1+count) * style_to_feats_dfreq[style] if style in style_to_feats_dfreq else 0 for style, count in style_counts.items()
#         } #TF-IDF
        
#         return style_distribution

#     def create_tfidf_rep(tfidf_dist, num_feats):
#         style_feats = sorted(tfidf_dist.items(), key=lambda x: -x[1])
#         top_k_feats = [x[0] for x in style_feats[:num_feats] if str(x[0]) != 'nan']
#         return top_k_feats

#     clusterd_df[output_clm +'_dist'] = clusterd_df.apply(lambda row: compute_tfidf(row), axis=1)
#     clusterd_df[output_clm]         = clusterd_df[output_clm +'_dist'].apply(lambda dist: create_tfidf_rep(dist, num_feats))

        
#     return clusterd_df

def compute_predicted_author(task_authors_df: pd.DataFrame, col_name: str) -> int:
    """
    Computes the predicted author based on the style features.
    """
    print("Computing predicted author using embeddings...")

    # Extract LUAR embeddings from task authors dataframe
    mystery_embedding = np.array(task_authors_df.iloc[0][col_name]).reshape(1, -1)
    candidate_embeddings = np.array([
        task_authors_df.iloc[1][col_name],
        task_authors_df.iloc[2][col_name],
        task_authors_df.iloc[3][col_name]
    ])

    # Compute cosine similarities
    similarities = cosine_similarity(mystery_embedding, candidate_embeddings)[0]
    predicted_author = int(np.argmax(similarities))
    print(f"Predicted author is Candidate {predicted_author + 1}")

    return predicted_author


def compute_precomputed_regions(bg_proj, bg_ids, q_proj, c_proj, pred_idx, model_name, n_neighbors=7):
    """
    Compute precomputed regions for mystery author and candidates.
    
    Args:
        bg_proj: (N,2) numpy array with 2D coordinates of background authors
        bg_ids: list of N author IDs for background authors
        q_proj: (1,2) numpy array with mystery author coordinates 
        c_proj: (3,2) numpy array with candidate author coordinates
        n_neighbors: number of closest neighbors to include in each region
    
    Returns:
        dict: mapping region names to bounding boxes and author lists
    """
    print("Computing sugested regions for zoom...")
    key = f"{hashlib.md5((model_name + str(q_proj.tolist()) + str(c_proj.tolist()) + str(n_neighbors)).encode()).hexdigest()}"
    
    if os.path.exists(REGION_CACHE):
        with open(REGION_CACHE, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}
    if key in cache:
        print(f"\n\n\n\033[1m\033[92m>>> Cache hit for {key} in {REGION_CACHE}: Using cached regions<<<\033[0m\n")
        return cache[key]
    else:
        print(f"\n\n\n\033[1m\033[91m>>> Cache miss for {key} in {REGION_CACHE}: Computing Regions<<<\033[0m\n")

    regions = {}
    
    # All points for distance calculation (mystery + candidates + background)
    all_points = np.vstack([q_proj.reshape(1, -1), c_proj, bg_proj])
    all_ids = ['mystery'] + [f'candidate_{i}' for i in range(3)] + bg_ids
    
    def get_region_around_point(center_point, region_name, include_points=None):
        """Get region around a specific point"""
        # Ensure center_point is 2D for euclidean_distances
        if center_point.ndim == 1:
            center_point = center_point.reshape(1, -1)
        
        # Calculate distances from center point to all background authors
        distances = euclidean_distances(center_point, bg_proj)[0]
        
        # Get indices of closest neighbors
        closest_indices = np.argsort(distances)[:n_neighbors]
        closest_authors = [bg_ids[i] for i in closest_indices]
        closest_points = bg_proj[closest_indices]
        
        # Include the center point in the region
        # region_points = np.vstack([center_point.reshape(1, -1), closest_points])
        if include_points is not None:
            region_points = include_points.copy()
            # Add center point and closest background authors
            region_points = np.vstack([region_points, center_point, closest_points])
        else:
            # Standard case - just center point and neighbors
            region_points = np.vstack([center_point, closest_points])
        
        # Calculate bounding box with some padding
        x_min, x_max = region_points[:, 0].min(), region_points[:, 0].max()
        y_min, y_max = region_points[:, 1].min(), region_points[:, 1].max()
        
        # Add padding (10% of range)
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        bbox = {
            'xaxis': [x_min - x_padding, x_max + x_padding],
            'yaxis': [y_min - y_padding, y_max + y_padding]
        }
        
        return {
            'bbox': bbox,
            'authors': closest_authors,
            'center_point': center_point,
            'description': f"Region around {region_name} ({len(closest_authors)} closest authors)"
        }
    
    def get_region_between_points(point1, point2, name1, name2):
        """Get region around the midpoint between two points"""
        midpoint = (point1 + point2) / 2
        region_name = f"{name1} & {name2}"
        # Include both original points in the region
        include_points = np.vstack([point1.reshape(1, -1), point2.reshape(1, -1)])
        return get_region_around_point(midpoint, region_name, include_points=include_points)

    # # Region 1: Around mystery author only
    # regions["Mystery Author Neighborhood"] = get_region_around_point(
    #     q_proj, "Mystery Author"
    # )
    
    # # Regions 2-4: Around each candidate
    for i in range(3):
        regions[f"Candidate {i+1} Neighborhood"] = get_region_around_point(
            c_proj[i], f"Candidate {i+1}"
        )
    
    # Regions 5-7: Between mystery and each candidate  
    for i in range(3):
        if i == pred_idx: #selecting only mystery and predicted candidate
            region_name = f"Mystery & Candidate {i+1}"
            regions[region_name] = get_region_between_points(
                q_proj, c_proj[i], "Mystery", f"Candidate {i+1}"
            )
    
    # Regions 8-10: Between candidate pairs
    # candidate_pairs = [(0, 1), (0, 2), (1, 2)]
    # for i, (c1, c2) in enumerate(candidate_pairs):
    #     if c1 != pred_idx and c2 != pred_idx: #selecting only the non predicated candidates
    #         region_name = f"Candidate {c1+1} & Candidate {c2+1}"
    #         regions[region_name] = get_region_between_points(
    #             c_proj[c1], c_proj[c2], f"Candidate {c1+1}", f"Candidate {c2+1}"
    #         )
    
    # Regions 11-12: Around predicted and ground truth (if different)
    # This would need predicted_author and ground_truth_author indices
    # For now, we'll create generic regions
    
    # Region 11: Centroid of all task authors (mystery + 3 candidates)
    # task_centroid = np.mean(np.vstack([q_proj, c_proj]), axis=0)
    # regions["All Task Authors Centroid"] = get_region_around_point(
    #     task_centroid, "All Task Authors", include_points=np.vstack([q_proj, c_proj])
    # )
    
    def serialize_numpy_dtypes(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: serialize_numpy_dtypes(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [serialize_numpy_dtypes(item) for item in obj]
        else:
            return obj

    serializable_regions = serialize_numpy_dtypes(regions)
    response = json.dumps(serializable_regions, default=str)
    cache[key] = response
    with open(REGION_CACHE, 'wb') as f: 
        print(f"\n\n\n\033[1m\033[92m>>> Cache saved for {key} in {REGION_CACHE} <<<\033[0m\n")
        pickle.dump(cache, f)

    return response

# if __name__ == "__main__":
#     background_corpus = pd.read_pickle('../datasets/luar_interp_space_cluster_19/train_authors.pkl')
#     print(background_corpus.columns)
#     print(background_corpus[['authorID', 'fullText', 'cluster_label']].head())
#     # # Example: Find features for clusters [2,3,4] that are NOT prominent in cluster [1]
#     # feats = compute_clusters_style_representation(
#     #     background_corpus_df=background_corpus,
#     #     cluster_ids=['00005a5c-5c06-3a36-37f9-53c6422a31d8',],
#     #     other_cluster_ids=[], # Pass the contrastive cluster IDs here
#     #     cluster_label_clm_name='authorID',
#     #     features_clm_name='final_attribute_name'
#     # )
#     # print(feats)
#     generate_style_embedding(background_corpus, 'fullText', 'AnnaWegmann/Style-Embedding')
#     print(background_corpus.columns)
