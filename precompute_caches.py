import ast
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import yaml

CONFIG_PATH="config/config.yaml"

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
# Load configuration and instances EXACTLY like app.py
cfg = load_config(CONFIG_PATH)
print(f"Configuration loaded from {CONFIG_PATH}")
print(f"config : \n{cfg}")

# Import your actual modules exactly as app.py does
from utils.file_download import download_file_override

download_file_override(cfg.get('background_authors_df_url'), cfg.get('background_authors_df_path'))
download_file_override(cfg.get('instances_to_explain_url'), cfg.get('instances_to_explain_path'))
download_file_override(cfg.get('gram2vec_feats_url'), cfg.get('gram2vec_feats_path'))

from utils.visualizations import get_instances, trigger_precomputed_region, handle_zoom_with_retries
from utils.ui import update_task_display

def precompute_all_caches(
    models_to_test=None,
    instances_to_process=None,
):
    """
    Precompute all cache files using the EXACT same methods as app.py.
    This follows the exact flow: load_task → update_task_display → run_visualization
    """
    
    if models_to_test is None:
        models_to_test = [
            'gabrielloiseau/LUAR-MUD-sentence-transformers',
            'gabrielloiseau/LUAR-CRUD-sentence-transformers', 
            'miladalsh/light-luar',
            'AnnaWegmann/Style-Embedding'
        ]
    
    print("\n\n" + "=" * 60)
    print("CACHE PRECOMPUTATION STARTED")
    print(f"Timestamp: {datetime.now()}")
    print(f"Models to test: {len(models_to_test)}")
    print("=" * 60)

    instances, instance_ids = get_instances(cfg['instances_to_explain_path'])
    # interp = load_interp_space(cfg)
    # clustered_authors_df = interp['clustered_authors_df']
    clustered_authors_df = pickle.load(open(cfg['background_authors_df_path'], 'rb'))
    
    if instances_to_process is None:
        instances_to_process = instance_ids
    
    print(f"Processing {len(instances_to_process)} instances with {len(models_to_test)} models")
    
    total_combinations = len(models_to_test) * len(instances_to_process)
    current_combination = 0
    
    cache_stats = {
        'embeddings_generated': 0,
        'tsne_computed': 0,
        'regions_computed': 0,
        'errors': []
    }
    
    for model_name in models_to_test:
        print(f"\n{'=' * 40}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'=' * 40}")
        
        for instance_id in tqdm(instances_to_process, desc=f"Processing instances for {model_name.split('/')[-1]}"):
            current_combination += 1
            try:
                # print(f"\n\n\n[{current_combination}/{total_combinations}] Processing Instance {instance_id}")
                print(f"\n\n\n\033[1m\033[93m>>> [{current_combination}/{total_combinations}] Processing Instance {instance_id} <<<\033[0m\n")

                
                # STEP 1: Replicate the exact flow from load_button.click()
                print("  → Replicating load_button.click() flow...")
                
                # Create ground truth (using placeholder since we're caching)
                ground_truth_author = None  # Will be determined by the instance data
                
                # Call update_task_display EXACTLY like app.py does
                task_results = update_task_display(
                    mode="Predefined Reddit Task",  # Always use predefined for caching
                    iid=f"Task {instance_id}",
                    instances=instances,
                    background_df=clustered_authors_df,
                    mystery_file=None,  # Not used for predefined
                    cand1_file=None,   # Not used for predefined  
                    cand2_file=None,   # Not used for predefined
                    cand3_file=None,   # Not used for predefined
                    true_author=ground_truth_author,
                    model_radio=model_name,
                    custom_model_input=""
                )
                
                # Extract the results exactly like app.py expects
                (header_html, mystery_html, c0_html, c1_html, c2_html, 
                 mystery_state, c0_state, c1_state, c2_state, 
                 task_authors_embeddings_df, background_authors_embeddings_df, 
                 predicted_author, ground_truth_author) = task_results
                
                print(f"    ✓ Embeddings generated for {len(task_authors_embeddings_df)} task authors")
                print(f"    ✓ Background embeddings: {len(background_authors_embeddings_df)} authors")
                cache_stats['embeddings_generated'] += 1
                
                # STEP 2: Replicate the exact flow from run_btn.click()
                print("  → Replicating run_btn.click() flow...")
                
                # Call visualize_clusters_plotly EXACTLY like app.py does
                viz_results = visualize_clusters_plotly(
                    iid=int(instance_id),
                    cfg=cfg,
                    instances=instances,
                    model_radio=model_name,
                    custom_model_input="",
                    task_authors_df=task_authors_embeddings_df,
                    background_authors_embeddings_df=background_authors_embeddings_df,
                    pred_idx=predicted_author,
                    gt_idx=ground_truth_author
                )
                
                # Extract results exactly like app.py expects
                (fig, style_names, bg_proj, bg_ids, bg_authors_df, 
                 precomputed_regions_state, precomputed_regions_radio) = viz_results
                
                print(f"    ✓ t-SNE projection computed")
                print(f"    ✓ Precomputed regions generated")
                cache_stats['tsne_computed'] += 1
                cache_stats['regions_computed'] += 1
                
                print(f"  ✓ Instance {instance_id} with model {model_name} completed successfully")
                

                print("  → Testing region zoom simulation...")
                if precomputed_regions_state:
                    regions_dict = ast.literal_eval(precomputed_regions_state)
                    test_regions = list(regions_dict.keys()) 
                    print(f"    → Found {len(test_regions)} regions to test")
                    
                    for region_name in test_regions:
                        try:
                            print(f"    → Testing region: {region_name}")
                            
                            # Step 3a: Simulate region selection (trigger_precomputed_region)
                            zoom_payload = trigger_precomputed_region(region_name, regions_dict)
                            
                            if zoom_payload:  # Only proceed if we got a valid zoom payload
                                # Step 3b: Simulate axis_ranges.change() (handle_zoom_with_retries)
                                zoom_results = handle_zoom_with_retries(
                                    event_json=zoom_payload,
                                    bg_proj=bg_proj,
                                    bg_lbls=bg_ids,
                                    clustered_authors_df=background_authors_embeddings_df,
                                    task_authors_df=task_authors_embeddings_df
                                )
                                
                                # Extract results like app.py does
                                (features_rb_update, gram2vec_rb_update, llm_style_feats_analysis, 
                                feature_list_state, visible_zoomed_authors) = zoom_results
                                
                                print(f"      ✓ LLM features cached for region: {region_name}")
                                
                        except Exception as e:
                            print(f"      ✗ Failed to cache features for region {region_name}: {e}")
                            # Continue with other regions even if one fails
                            continue
            except Exception as e:
                error_msg = f"Error processing instance {instance_id} with model {model_name}: {str(e)}"
                print(f"  ✗ {error_msg}")
                cache_stats['errors'].append(error_msg)
                import traceback
                traceback.print_exc()
                continue
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("CACHE PRECOMPUTATION COMPLETED")
    print("=" * 60)
    print(f"Embeddings generated: {cache_stats['embeddings_generated']}")
    print(f"t-SNE projections computed: {cache_stats['tsne_computed']}")
    print(f"Region sets computed: {cache_stats['regions_computed']}")
    print(f"Errors encountered: {len(cache_stats['errors'])}")
    
    if cache_stats['errors']:
        print("\nERROR DETAILS:")
        for error in cache_stats['errors']:
            print(f"  - {error}")
    
    return cache_stats

# Import the exact functions your app uses
from utils.visualizations import visualize_clusters_plotly

if __name__ == "__main__":
    # Test with a small subset first
    instances=[i for i in range(20)]  # First 10 instances for testing
    cache_stats = precompute_all_caches(
        models_to_test=[
             'AnnaWegmann/Style-Embedding'
        ],
        instances_to_process=instances
    )
    
    print(f"\nCache precomputation completed with {len(cache_stats['errors'])} errors.")