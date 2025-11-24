import argparse
import pandas as pd
import numpy as np
import os

from utils.interp_space_utils import cached_generate_style_embedding
from utils.clustering_utils import clustering_author

def load_corpus(filepath: str) -> pd.DataFrame:
    """
    Loads a corpus from a CSV or Pickle file into a pandas DataFrame.
    The file is expected to have 'authorID' and 'fullText' columns.
    """
    print(f"Loading corpus from {filepath}...")
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.pkl'):
        df = pd.read_pickle(filepath)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .pkl")

    if 'authorID' not in df.columns or 'fullText' not in df.columns:
        raise ValueError("Corpus must contain 'authorID' and 'fullText' columns.")

    print(f"Corpus loaded successfully with {len(df)} documents.")
    return df

def main():
    """
    Main function to run the clustering workflow.
    """
    parser = argparse.ArgumentParser(
        description="Generate style embeddings and cluster a corpus of documents."
    )
    parser.add_argument(
        "corpus_path",
        type=str,
        help="Path to the corpus file (.csv or .pkl)."
    )
    parser.add_argument(
        "test_corpus_path",
        type=str,
        help="Path to the test corpus file (.csv or .pkl)."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Hugging Face model name for sentence-transformer embeddings (e.g., 'AnnaWegmann/Style-Embedding')."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the output DataFrame with embeddings and clusters (.pkl)."
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="min_samples parameter for DBSCAN clustering."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='cosine',
        choices=['cosine', 'euclidean'],
        help="Distance metric for DBSCAN clustering."
    )
    parser.add_argument(
        "--eps_values",
        type=float,
        nargs='+',
        default=None,
        help="A list of specific eps values to test for DBSCAN. If not provided, a default range is used."
    )
    parser.add_argument(
        "--pca_dimensions",
        type=int,
        default=None,
        help="If provided, apply PCA to reduce embeddings to this number of dimensions before clustering."
    )

    args = parser.parse_args()

    # 1. Load the corpus
    corpus_df = load_corpus(args.corpus_path)
    test_corpus_df = load_corpus(args.test_corpus_path)

    #print(corpus_df)
    # 2. Generate style embeddings
    print(f"\nGenerating style embeddings with model: {args.model_name}")
    # The function returns two dataframes, we are only interested in the first one here.
    # We pass `task_authors_df=None` as we are processing a single corpus.
    clustered_df, _ = cached_generate_style_embedding(
        background_corpus_df=corpus_df,
        text_clm='fullText',
        model_name=args.model_name,
        task_authors_df=None
    )

    clustered_test_df, _ = cached_generate_style_embedding(
        background_corpus_df=test_corpus_df,
        text_clm='fullText',
        model_name=args.model_name,
        task_authors_df=None
    )
    embedding_col_name = f'{args.model_name.split("/")[-1]}_style_embedding'
    print(f"Embeddings generated and stored in column '{embedding_col_name}'.")

    # 3. Perform clustering
    print(f"\nPerforming DBSCAN clustering with metric='{args.metric}' and min_samples={args.min_samples}...")
    clustered_df = clustering_author(
        background_corpus_df=clustered_df,
        test_corpus_df=clustered_test_df,
        embedding_clm=embedding_col_name,
        eps_values=args.eps_values,
        min_samples=args.min_samples,
        pca_dimensions=args.pca_dimensions,
        metric=args.metric
    )
    
    # remove authors with cluster label == -1
    clustered_df = clustered_df[clustered_df['cluster_label'] != -1]

    # 4. Save the results
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    clustered_df.to_pickle(args.output_path)
    print(f"\nSuccessfully saved clustered DataFrame to: {args.output_path}")
    print(f"DataFrame includes cluster labels in the 'cluster_label' column.")

if __name__ == "__main__":
    main()