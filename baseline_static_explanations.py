import argparse
import pandas as pd
import numpy as np
import os, json

from utils.interp_space_utils import cached_generate_style_embedding
from utils.clustering_utils import clustering_author
from utils.interp_space_utils import compute_clusters_style_representation_3, summarize_style_features_to_paragraph, find_closest_cluster_style

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


def build_static_interp_space(cluster_df):
    """
    Takes a dataframe with cluster_label indicates every author's cluster and return a
    json file with key the cluster_label and value containing the style-embedding representation and the style description

    Example cluster_df
                                                 fullText         authorID                    Style-Embedding_style_embedding  cluster_label
        4   [I've play them all (D3, Torchlight 1&amp;2, P...         HaxRyter  [0.7126333904811682, -0.5076461933032986, -0.1...              0
        10  [Back in Texas.  Buddy had a kid in an up and ...  OaklandHellBent  [0.11238726238181786, 0.9263576185812101, -0.2...              1

    """
    # Find the embedding column (assuming it's the only one ending with '_style_embedding')
    embedding_clm = next((col for col in cluster_df.columns if col.endswith('_style_embedding')), None)
    if not embedding_clm:
        raise ValueError("No style embedding column found in the DataFrame.")

    print(f"Using embedding column: {embedding_clm}")

    # Group by cluster label and calculate the average embedding for each cluster
    # We also aggregate authorIDs to use them for style representation
    cluster_groups = cluster_df.groupby('cluster_label').agg({
        embedding_clm: lambda embs: np.mean(np.vstack(embs), axis=0).tolist(),
        'authorID': list
    }).reset_index()

    interpretable_space = {}

    for _, row in cluster_groups.iterrows():
        cluster_label = row['cluster_label']
        avg_embedding = row[embedding_clm]
        author_ids_in_cluster = row['authorID']

        print(f"\nProcessing cluster {cluster_label} with {len(author_ids_in_cluster)} authors...")

        # Generate style description using an LLM
        # We reuse the utility function from the interactive tool for consistency
        style_analysis = compute_clusters_style_representation_3(
            background_corpus_df=cluster_df,
            cluster_ids=author_ids_in_cluster,
            cluster_label_clm_name='authorID',
            max_num_feats=5, # Requesting 5 top features
            max_num_authors=20, # Use up to 20 authors from the cluster for analysis
            return_only_feats=True
        )

        # When return_only_feats=True, style_analysis is a list of features
        style_features_list = style_analysis
        print(f"  Generated style features: {style_features_list}")

        # Summarize the list of features into a coherent paragraph
        style_paragraph = summarize_style_features_to_paragraph(style_features_list)
        print(f"  Summarized paragraph: {style_paragraph}")

        # JSON cannot serialize numpy integers, so convert cluster_label
        interpretable_space[int(cluster_label)] = (avg_embedding, style_paragraph)
    
    return interpretable_space

def generate_explanations(args):
    input_file = args.input_file
    interp_space_path = args.interp_space_path
    output_file = args.output_file 
    model_name  = args.model_name if args.model_name else 'AnnaWegmann/Style-Embedding'

    instances_for_ex = json.load(open(input_file))
    interp_space = json.load(open(interp_space_path))

    output = []
    for instance in instances_for_ex:
        json_obj = {}
        json_obj['Q_authorID'] = instance['Q_authorID']
        json_obj['Q_fullText'] = '\n\n'.join(instance['Q_fullText'])
        style_descirption, q_embeddings = find_closest_cluster_style(instance['Q_fullText'], interp_space, model_name=model_name)
        json_obj['Q_top_style_feats'] = style_descirption
        
        json_obj['a0_authorID'] = instance['a0_authorID']
        json_obj['a0_fullText'] = '\n\n'.join(instance['a0_fullText'])
        style_descirption, a0_embeddings = find_closest_cluster_style(instance['a0_fullText'], interp_space, model_name=model_name)
        json_obj['a0_top_style_feats'] = style_descirption

        json_obj['a1_authorID'] = instance['a1_authorID']
        json_obj['a1_fullText'] = '\n\n'.join(instance['a1_fullText'])
        style_descirption, a1_embeddings = find_closest_cluster_style(instance['a1_fullText'], interp_space, model_name=model_name)
        json_obj['a1_top_style_feats'] = style_descirption

        json_obj['a2_authorID'] = instance['a2_authorID']
        json_obj['a2_fullText'] = '\n\n'.join(instance['a2_fullText'])
        style_descirption, a2_embeddings = find_closest_cluster_style(instance['a2_fullText'], interp_space, model_name=model_name)
        json_obj['a2_top_style_feats'] = style_descirption

        json_obj['gt_idx'] = instance['gt_idx']
        
        # Compute pairwise similarity between q_embeddings and all a_embeddings
        # Ensure embeddings are 2D arrays for cosine_similarity
        q_emb_2d  = np.array(q_embeddings).reshape(1, -1)
        a0_emb_2d = np.array(a0_embeddings).reshape(1, -1)
        a1_emb_2d = np.array(a1_embeddings).reshape(1, -1)
        a2_emb_2d = np.array(a2_embeddings).reshape(1, -1)

        similarity_q_a0 = cosine_similarity(q_emb_2d, a0_emb_2d)[0][0]
        similarity_q_a1 = cosine_similarity(q_emb_2d, a1_emb_2d)[0][0]
        similarity_q_a2 = cosine_similarity(q_emb_2d, a2_emb_2d)[0][0]

        ranked_candidates = [
            {'authorID': instance['a0_authorID'], 'similarity': float(similarity_q_a0)},
            {'authorID': instance['a1_authorID'], 'similarity': float(similarity_q_a1)},
            {'authorID': instance['a2_authorID'], 'similarity': float(similarity_q_a2)},
        ]

        json_obj['latent_rank'] = np.argsort([x['similarity'] for x in ranked_candidates]).tolist()
        json_obj['model_pred'] = 'Candidate {}'.format(json_obj['latent_rank'][0] + 1)



        output.append(json_obj)

    json.dump(output, open(output_file, 'w'), indent=4)




def main():
    """
    Main function to generate and save the static interpretable space.
    """

    parser = argparse.ArgumentParser(
        description="Build a static interpretable space from clustered author data."
    )
    
    parser.add_argument(
        "task",
        type=str,
        help="task: one of the following: build_static_interp_space, generate_explanations",
        choices=["build_static_interp_space", "generate_explanations"]
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input clustered DataFrame (.pkl file)."
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="file to save the output"
    )

    parser.add_argument(
        "--interp_space_path",
        type=str,
        help="Path to the input interpretable space(.pkl file)."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="style analysis model name"
    )

    args = parser.parse_args()

    if args.task == "build_static_interp_space":
        return build_and_save_static_interp_space(args)
    elif args.task == "generate_explanations":
        return generate_explanations(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")


def build_and_save_static_interp_space(args):
    print(f"Loading clustered data from {args.input_file}...")
    clustered_df = pd.read_pickle(args.input_file)

    interpretable_space = build_static_interp_space(clustered_df)

    print(f"\nSaving interpretable space to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(interpretable_space, f, indent=4)
    
    print("Done.")

if __name__ == "__main__":
    main()
