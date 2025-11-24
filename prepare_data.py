import json
import argparse
import csv
import sys
import copy
import os 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import minmax_scale
import random


import pickle

import json
import pandas as pd

def sample_ds(input_file, output_file, num_insts=10000, min_num_text_per_inst=0, max_num_text_per_inst=3):
    """
    sample_ds('/mnt/swordfish-pool2/nikhil/raw_all/test_queries.jsonl', '/mnt/swordfish-pool2/milad/hiatus-data/reddit_cluster_test.pkl', 
          num_insts=10000, 
          min_num_text_per_inst=3, 
          max_num_text_per_inst=10)

    sample_ds('/mnt/swordfish-pool2/nikhil/raw_all/data.jsonl', '/mnt/swordfish-pool2/milad/hiatus-data/reddit_cluster_training.pkl', 
          num_insts=10000, 
          min_num_text_per_inst=3, 
          max_num_text_per_inst=10)
    """
    f = open(input_file)
    out_list = []
    for i in range(num_insts):    
        json_obj = json.loads(f.readline())
        if len(json_obj['syms']) < min_num_text_per_inst:
            continue
            
        out_list.append({
            'fullText': json_obj['syms'][:max_num_text_per_inst],
            'authorID': json_obj['author_id']
        })
    df = pd.DataFrame(out_list)
    df.to_pickle(output_file)

    df = df.explode('fullText').reset_index()
    df.to_json(output_file.replace('.pkl', '.json'))

def get_reddit_data(input_path, random_seed=123, num_instances=100, num_documents_per_author=8, min_instance_len=10):
    
    df = pd.read_pickle(open(input_path, 'rb'))
    df['fullText'] = df.fullText.map(lambda x: [d for d in x if len(d.split()) > min_instance_len])
    df = df[df.fullText.str.len() > num_documents_per_author * 2]
    
    output_objs = []

    for _, row in df.iterrows():

        # Get the current author's documents
        query_author_df   = df[df.authorID == row['authorID']]
        # split the author's documents into two: query and correct author
        author_documents = [x for x in query_author_df.fullText.tolist()[0] if len(x.split()) > min_instance_len]

        if len(author_documents) <= num_documents_per_author * 2:
            continue

        query_documents    = author_documents[:num_documents_per_author]
        correct_documents  = author_documents[num_documents_per_author:]

        
        # Sample two *other* authors
        other_authors_df = df[df.authorID != row['authorID']]
        other_two_authors = other_authors_df.sample(2, random_state=random_seed)
        
        # Make sure all authors are are of equivelant number of texts
        min_found_texts = min([len(correct_documents), len(query_documents)] + [len(x) for x in other_two_authors.fullText.tolist()])
        query_documents = query_documents[:min_found_texts]
        correct_documents = correct_documents[:min_found_texts]
        other_two_authors.fullText = other_two_authors.fullText.apply(lambda x: x[:min_found_texts])
        
        output_objs.append({
            "Q_authorID":  str(row["authorID"]),
            "Q_fullText":  ["Text:\n{}".format(d) for d in query_documents],
            "a0_authorID":  str(other_two_authors.iloc[0]["authorID"]),
            "a0_fullText": ["Text:\n{}".format(d) for d in other_two_authors.iloc[0]["fullText"][:num_documents_per_author]],
            "a1_authorID":  str(other_two_authors.iloc[1]["authorID"]),
            "a1_fullText": ["Text:\n{}".format(d) for d in other_two_authors.iloc[1]["fullText"][:num_documents_per_author]],
            "a2_authorID": str(row["authorID"]) + "_correct",
            "a2_fullText": ["Text:\n{}".format(d) for d in correct_documents],
            "gt_idx": 2
        })
        print( "Text:\n\n".join(query_documents))
        random_seed += 1 # Increment seed to get different authors for the next task
        if len(output_objs) >= num_instances:
            break

    return output_objs


def get_iarapa_pilot_data(input_path):
    for data_point in glob.glob(input_path + '*/'):
        candidates_file = list(glob.glob(data_point + '/data/*_candidates.jsonl'))[0]
        queries_file    = list(glob.glob(data_point + '/data/*_queries.jsonl'))[0]
        grount_truth_file = list(glob.glob(data_point + '/groundtruth/*_groundtruth.npy'))[0]
        q_labels_file = glob.glob(data_point + '/groundtruth/*_query-labels.txt')[0]
        c_labels_file = glob.glob(data_point + '/groundtruth/*_candidate-labels.txt')[0]
        
        candidates_df = pd.read_json(candidates_file, lines=True)
        queries_df = pd.read_json(queries_file, lines=True)
        
        queries_df['authorID'] = queries_df.authorIDs.apply(lambda x: x[0])
        candidates_df['authorID'] = candidates_df.authorSetIDs.apply(lambda x: x[0])
        
        queries_df = queries_df.groupby('authorID').agg({'fullText': lambda x: list(x)}).reset_index()
        candidates_df = candidates_df.groupby('authorID').agg({'fullText': lambda x: list(x)}).reset_index()
            
        ground_truth_assignment = np.load(open(grount_truth_file, 'rb'))
        candidate_authors = [a[2:-3] for a in  open(c_labels_file).read().split('\n')][:-1]
        query_authors = [a[2:-3] for a in  open(q_labels_file).read().split('\n')][:-1]
    
        #print(ground_truth_assignment)
        #print(candidate_authors)
        #print(query_authors)
        yield query_authors, candidate_authors, queries_df, candidates_df, ground_truth_assignment

def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description="Prepare Reddit data for author attribution tasks.")
    parser.add_argument("input_path", type=str, help="Path to the input pandas DataFrame pickle file.")
    parser.add_argument("output_path", type=str, help="Path to save the output JSON file.")
    parser.add_argument("--random_seed", type=int, default=123, help="Random seed for sampling.")
    parser.add_argument("--num_docs", type=int, default=5, help="Number of documents per author for query and correct sets.")
    
    args = parser.parse_args()

    print(f"Processing data from: {args.input_path}")
    output_data = get_reddit_data(
        input_path=args.input_path,
        random_seed=args.random_seed,
        num_documents_per_author=args.num_docs
    )

    print(f"Saving {len(output_data)} tasks to: {args.output_path}")
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print("Done.")

if __name__ == "__main__":
    main()
