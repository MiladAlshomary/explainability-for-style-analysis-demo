import gradio as gr
import json
import numpy as np
from sklearn.manifold import TSNE
import pickle as pkl
import os
import hashlib
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from gradio import update
import re
from utils.interp_space_utils import compute_clusters_style_representation_3, compute_clusters_g2v_representation, compute_precomputed_regions
from utils.llm_feat_utils import split_features
from utils.gram2vec_feat_utils import get_shorthand, get_fullform
from gram2vec.feature_locator import find_feature_spans
import plotly.io as pio

def clean_text(text: str) -> str:
    """
    Cleans the text by replacing HTML tags with their escaped versions.
    """
    return text.replace('<','&lt;').replace('>','&gt;').replace('\n', '<br>')

def get_instances(instances_to_explain_path: str = 'datasets/instances_to_explain.json'):
    """
    Loads the JSON and returns:
      - instances_to_explain: the raw dict/list of instances
      - instance_ids: list of keys (if dict) or indices (if list)
    """
    instances_to_explain = json.load(open(instances_to_explain_path))
    if isinstance(instances_to_explain, dict):
        instance_ids = list(instances_to_explain.keys())
    else:
        instance_ids = list(range(len(instances_to_explain)))
    return instances_to_explain, instance_ids

def load_instance(instance_id, instances_to_explain: dict):
    """
    Given a selected instance_id and the loaded data,
    returns (mystery_html, c0_html, c1_html, c2_html).
    """
    # normalize instance_id
    try:
        iid = int(instance_id)
    except ValueError:
        iid = instance_id
    data = instances_to_explain[iid]

    predicted_author = data['latent_rank'][0]
    ground_truth_author = data['gt_idx']

    header_html = f"""
    <div style="border:1px solid #ccc; padding:10px; margin-bottom:10px;">
      <h3>Here’s the mystery passage alongside three candidate texts—look for the green highlight to see the predicted author.</h3>
    </div>
    """
    mystery_text = clean_text(data['Q_fullText'])
    mystery_html = f"""
    <div style="
            border: 2px solid #ff5722;      /* accent border */
            background: #fff3e0;            /* very light matching wash */
            border-radius: 6px;
            padding: 1em;
            margin-bottom: 1em;
        ">
        <h3 style="margin-top:0; color:#bf360c;">Mystery Author</h3>
        <p>{clean_text(mystery_text)}</p>
    </div>
    """

    # Candidate boxes
    candidate_htmls = []
    for i in range(3):
        text = data[f'a{i}_fullText']
        title = f"Candidate {i+1}"
        extra_style = ""

        if ground_truth_author == i:
            if ground_truth_author != predicted_author: # highlight the true author only if its different than the predictd one
                title += " (True Author)"
                extra_style = (
                    "border: 2px solid #ff5722; "
                    "background: #fff3e0; " 
                    "padding:10px; "
                )

        
        if predicted_author == i:
            if predicted_author == ground_truth_author:
                title += " (Predicted and True Author)"
            else:
                title += " (Predicted Author)"
            extra_style = (
                "border:2px solid #228B22; "        # dark green border
                "background-color: #e6ffe6; "       # light green fill
                "padding:10px; "
            )
            

        candidate_htmls.append(f"""
        <div style="border:1px solid #ccc; padding:10px; {extra_style}">
          <h4>{title}</h4>
          <p>{clean_text(text)}</p>
        </div>
        """)

    return header_html, mystery_html, candidate_htmls[0], candidate_htmls[1], candidate_htmls[2]

def compute_tsne_with_cache(embeddings: np.ndarray, cache_path: str = 'datasets/tsne_cache.pkl') -> np.ndarray:
    """
    Compute t-SNE with caching to avoid recomputation for the same input.

    Args:
        embeddings (np.ndarray): The input embeddings to compute t-SNE on.
        cache_path (str): Path to the cache file.

    Returns:
        np.ndarray: The t-SNE transformed embeddings.
    """
    # Create a hash of the input embeddings to use as a key
    hash_key = hashlib.md5(embeddings.tobytes()).hexdigest()
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pkl.load(f)
    else:
        cache = {}

    if hash_key in cache:
        return cache[hash_key]
    else:
        print("Computing t-SNE")
        tsne_result = TSNE(n_components=2, learning_rate='auto',
                          init='random', perplexity=10, random_state=42, metric='cosine').fit_transform(embeddings)
        #tsne_result = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, metric='cosine').fit_transform(embeddings)
        
        cache[hash_key] = tsne_result
        with open(cache_path, 'wb') as f:
            pkl.dump(cache, f)
        return tsne_result  

def load_interp_space(cfg):
    interp_space_path      = cfg['interp_space_path'] + 'interpretable_space.pkl'
    interp_space_rep_path  = cfg['interp_space_path'] + 'interpretable_space_representations.json'
    gram2vec_feats_path    = cfg['interp_space_path'] + '/../gram2vec_feats.csv'
    clustered_authors_path = cfg['interp_space_path'] + 'train_authors.pkl'

    max_num_docs_per_authors = cfg['max_num_docs_per_authors']
    max_num_bg_authors = cfg['max_num_bg_authors']

    # Load authors embeddings and their cluster labels
    clustered_authors_df = pd.read_pickle(clustered_authors_path).iloc[:max_num_bg_authors]
    clustered_authors_df['fullText'] = clustered_authors_df.fullText.map(lambda list: '\n\n'.join(['Document {}: {}'.format(i+1, text) for i, text in enumerate(list[:max_num_docs_per_authors])]))

    print('Average atuhor text length:', clustered_authors_df.fullText.map(lambda x: len(x.split())).mean())

    author_embedding = clustered_authors_df.author_embedding.tolist()
    author_labels    = clustered_authors_df.cluster_label.tolist()
    author_ids      = clustered_authors_df.authorID.tolist()

    # filter out gram2vec features that doesn't have representation
    clustered_authors_df['gram2vec_feats'] = clustered_authors_df.gram2vec_feats.apply(lambda feats: [feat for feat in feats if get_shorthand(feat) is not None])
    
    # Load a list of gram2vec features --> we use it to distinguish the cluster representations whether they come from gram2vec or llms
    gram2vec_df = pd.read_csv(gram2vec_feats_path)
    gram2vec_feats = gram2vec_df.gram2vec_feats.unique().tolist()

    # Load interpretable space embeddings and the representation of each dimension
    interpretable_space = pkl.load(open(interp_space_path, 'rb'))
    del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
    dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}

    interpretable_space_rep_df = pd.read_json(interp_space_rep_path)
    #dimension_to_style  = {x[0]: x[1] for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_clm].tolist())}
    dimension_to_style  = {x[0]: [feat[0] for feat in sorted(x[1].items(), key=lambda feat_w:-feat_w[1])] for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[cfg['style_feat_clm']].tolist())}

    if cfg['only_llm_feats']:
        #print('only llm feats')
        dimension_to_style = {dim[0]:[feat for feat in dim[1] if feat not in gram2vec_feats] for dim in dimension_to_style.items()}

    if cfg['only_gram2vec_feats']:
        #print('only gra2vec feats')
        dimension_to_style = {dim[0]:[feat for feat in dim[1] if feat in gram2vec_feats] for dim in dimension_to_style.items()}

    # Take top features from g2v and llm
    def take_to_k_llm_and_g2v_feats(feats_list, top_k):
        g2v_feats = [x for x in feats_list if x in gram2vec_feats][:top_k]
        llm_feats = [x for x in feats_list if x not in gram2vec_feats][:top_k]
        return g2v_feats + llm_feats
    dimension_to_style = {dim[0]: take_to_k_llm_and_g2v_feats(dim[1], cfg['top_k']) for dim in dimension_to_style.items()}


    return {
        'dimension_to_latent': dimension_to_latent, 
        'dimension_to_style' : dimension_to_style, 
        'author_embedding' : author_embedding, 
        'author_labels' : author_labels,
        'author_ids' : author_ids,
        'clustered_authors_df' : clustered_authors_df

    }

# Function to process G2V features and create display choices
def format_g2v_features_for_display(g2v_features_with_scores):
    """
    Convert G2V features into display format for Gradio radio buttons.
    
    Args:
        g2v_features_with_scores: List of tuples like:
            [('None', None), ('Feature Name', score), ...]
    
    Returns:
        tuple: (display_choices, original_values)
    """
    display_choices = []
    original_values = []
    
    for item in g2v_features_with_scores:
        if len(item) == 2:
            feature_name, score = item
            
            # Handle None case
            if feature_name == "None" or score is None:
                display_choices.append("None")
                original_values.append("None")
            else:
                # Just show the feature name without scores
                display_choices.append(feature_name)
                original_values.append(feature_name)
        else:
            # Handle unexpected format
            display_choices.append(str(item))
            original_values.append(str(item))
    
    return display_choices, original_values

#function to handle zoom events
def handle_zoom(event_json, bg_proj, bg_lbls, clustered_authors_df, task_authors_df, predicted_author=None):
    """
    event_json         – stringified JSON from JS listener
    bg_proj            – (N,2) numpy array with 2D coordinates
    bg_lbls            – list of N author IDs
    clustered_authors_df – pd.DataFrame containing authorID and final_attribute_name
    task_authors_df    – pd.DataFrame containing task authors
    predicted_author   – index of predicted author (0, 1, or 2)
    """
    print("[INFO] Handling zoom event")
    print(f"[INFO] Predicted author: {predicted_author}")

    if not event_json:
        return gr.update(value=""), gr.update(value=""), None, None, None

    try:
        ranges = json.loads(event_json)
        (x_min, x_max) = ranges["xaxis"]
        (y_min, y_max) = ranges["yaxis"]
    except (json.JSONDecodeError, KeyError, ValueError):
        return gr.update(value=""), gr.update(value=""), None, None, None

    # Find points within the zoomed region
    mask = (
        (bg_proj[:, 0] >= x_min) & (bg_proj[:, 0] <= x_max) &
        (bg_proj[:, 1] >= y_min) & (bg_proj[:, 1] <= y_max)
    )

    visible_authors = [lbl for lbl, keep in zip(bg_lbls, mask) if keep]

    print(f"[INFO] Zoomed region includes {len(visible_authors)} authors:{visible_authors}")

    print(f"Task authors: {len(task_authors_df)}, Clustered authors: {len(clustered_authors_df)}")
    merged_authors_df = pd.concat([task_authors_df, clustered_authors_df])
    print(f"Merged authors DataFrame:\n{len(merged_authors_df)}")
    #style_analysis_response = {'features': [], 'spans': []}
    style_analysis_response = compute_clusters_style_representation_3(
        background_corpus_df=merged_authors_df,
        cluster_ids=visible_authors,
        cluster_label_clm_name='authorID',
        predicted_author=predicted_author
    )

    llm_feats = ['None'] + style_analysis_response['features']


    merged_authors_df = pd.concat([task_authors_df, clustered_authors_df])
    #g2v_feats = []
    g2v_feats = compute_clusters_g2v_representation(
        background_corpus_df=merged_authors_df,
        author_ids=visible_authors,
        other_author_ids=[],
        features_clm_name='g2v_vector',
        top_n=15,
        predicted_author=predicted_author
    )

    # ── Span-existence filter on task authors in the zoom ───────────────────
    # Keep only features that have detected spans in at least 2 of the
    # task authors' texts (Mystery + Candidates 1-3)
    # Use only the task authors (Mystery + Candidates 1-3), not the zoom-visible set
    # task_author_ids = {"Mystery author", "Candidate Author 1", "Candidate Author 2", "Candidate Author 3"}
    # task_only_df = task_authors_df[task_authors_df['authorID'].isin(task_author_ids)]
    # if task_only_df.empty:
    #     task_only_df = task_authors_df

    # def _to_text(x):
    #     return '\n\n'.join(x) if isinstance(x, list) else x

    # task_texts = [_to_text(x) for x in task_only_df['fullText'].tolist()]

    # print(f"len task_texts: {len(task_texts)}")
    # filtered_g2v_feats = []
    # for feat in g2v_feats:
    #     try:
    #         # `feat` is shorthand already (e.g., 'pos_bigrams:NOUN PROPN')
    #         occurrences = 0
    #         for txt in task_texts:
    #             spans = find_feature_spans(txt, feat[0])
    #             if spans:
    #                 occurrences += 1
    #         if occurrences >= 2:
    #             filtered_g2v_feats.append(feat)
    #         else:
    #             print(f"[INFO] Dropping G2V feature with <2 task-author spans: {feat}")
    #     except Exception as e:
    #         print(f"[WARN] Error while checking spans for {feat}: {e}")

    # # After filtering by spans, keep top-N by score
    # filtered_g2v_feats = filtered_g2v_feats[:10]
    filtered_g2v_feats = g2v_feats

    # Convert to human readable for display
    HR_g2v_list = []
    for feat in filtered_g2v_feats:
        HR_g2v = get_fullform(feat[0])
        # print(f"\n\n feat: {feat} ---> Human Readable: {HR_g2v}")
        if HR_g2v is None:
            #print(f"Skipping Gram2Vec feature without human readable form: {feat}")
            HR_g2v_list.append((feat[0], feat[1])) #get the score
        else:
            HR_g2v_list.append((HR_g2v, feat[1])) #get the score

    HR_g2v_list = [("None", None)] + HR_g2v_list

    print(f"[INFO] Found {len(llm_feats)} LLM features and {len(g2v_feats)} Gram2Vec features in the zoomed region.")   
    # print(f"[INFO] unfiltered g2v features: {g2v_feats}")

    print(f"[INFO] LLM features: {llm_feats}")
    HR_g2v_list, _ = format_g2v_features_for_display(HR_g2v_list)
    # print(f"[INFO] Gram2Vec features: {HR_g2v_list}")

    return (
        gr.update(choices=llm_feats, value=llm_feats[0]),
        gr.update(choices=HR_g2v_list, value=HR_g2v_list[0]),
        style_analysis_response,
        llm_feats,
        visible_authors
    )
    # return gr.update(value="\n".join(llm_feats).join("\n").join(g2v_feats)), llm_feats, g2v_feats

def handle_zoom_with_retries(event_json, bg_proj, bg_lbls, clustered_authors_df, task_authors_df, predicted_author=None):
    """
    event_json         – stringified JSON from JS listener
    bg_proj            – (N,2) numpy array with 2D coordinates
    bg_lbls            – list of N author IDs
    clustered_authors_df – pd.DataFrame containing authorID and final_attribute_name
    task_authors_df   – pd.DataFrame containing authorID and final_attribute_name
    predicted_author  – index of predicted author (0, 1, or 2)
    """
    print("[INFO] Handling zoom event with retries")

    for attempt in range(3):
        try:
            return handle_zoom(event_json, bg_proj, bg_lbls, clustered_authors_df, task_authors_df, predicted_author)
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print("[INFO] Retrying...")
    return (
        None,
        None,
        None,
        None,
        None
    )


def visualize_clusters_plotly(iid, cfg, instances, model_radio, custom_model_input, task_authors_df, background_authors_embeddings_df, pred_idx=None, gt_idx=None):
    model_name = model_radio if model_radio != "Other" else custom_model_input
    embedding_col_name = f'{model_name.split("/")[-1]}_style_embedding'
    # print(background_authors_embeddings_df.columns)
    print("Generating cluster visualization")
    iid = int(iid)
    #interp      = load_interp_space(cfg)
    # dim2lat     = interp['dimension_to_latent']
    #style_names = interp['dimension_to_style']
    # bg_emb      = np.array(interp['author_embedding'])
    # print(f"bg_emb shape: {bg_emb.shape}")
    #replace with cached embedddings
    bg_emb      = np.array(background_authors_embeddings_df[embedding_col_name].tolist()) #placeholder for background embeddings
    print(f"bg_emb shape: {bg_emb.shape}")
    # print("interp.keys():", interp.keys())
    #bg_lbls     = interp['author_labels']
    #bg_ids      = interp['author_ids']
    bg_ids = task_authors_df['authorID'].tolist() + background_authors_embeddings_df['authorID'].tolist()
    # inst         = instances[iid]
    # print("inst.keys():", inst.keys())
    # q_lat        = np.array(inst['author_latents'][:1])
    # print(f"q_lat shape: {q_lat.shape}")
    # c_lat        = np.array(inst['author_latents'][1:])
    # print(f"c_lat shape: {c_lat.shape}")
    # pred_idx     = inst['latent_rank'][0]
    # gt_idx       = inst['gt_idx']
    q_lat = np.array(task_authors_df[embedding_col_name].iloc[0]).reshape(1, -1) # Mystery author latent
    print(f"q_lat shape: {q_lat.shape}")
    c_lat = np.array(task_authors_df[embedding_col_name].iloc[1:].tolist())  # Candidate authors latents
    print(f"c_lat shape: {c_lat.shape}")

    # cent_emb = np.array([v for _,v in dim2lat.items()])
    # cent_lbl = np.array([k for k,_ in dim2lat.items()])

    # all_emb = np.vstack([q_lat, c_lat, bg_emb, cent_emb])
    all_emb = np.vstack([q_lat, c_lat, bg_emb])
    proj    = compute_tsne_with_cache(all_emb)

    # split
    q_proj    = proj[0]
    c_proj    = proj[1:4]
    bg_proj   = proj


    # 2) build Plotly figure
    fig = go.Figure()

    fig.update_layout(
        template='plotly_white',
        margin=dict(l=40,r=40,t=60,b=40),
        autosize=True,
        hovermode='closest',
        # Enable zoom events
        dragmode='zoom'  
    )

    # uncomment the following line to show background authors
    ## background authors (light grey dots)
    fig.add_trace(go.Scattergl(
        x=bg_proj[:,0], y=bg_proj[:,1],
        mode='markers',
        marker=dict(size=6, color="#d3d3d3"),# color=bg_colors
        name='Background authors',
        hoverinfo='skip'
    ))

    # three candidates
    marker_syms = ['diamond','pentagon','x']
    for i in range(3):
        # label = f"Candidate {i+1}" + (" (predicted)" if i==pred_idx else "")
        base = f"Candidate {i+1}"
        # pick the right suffix
        if i == pred_idx and i == gt_idx:
            suffix = " (Predicted & Ground Truth)"
        elif i == pred_idx:
            suffix = " (Predicted)"
        elif i == gt_idx:
            suffix = "(Ground Truth)"
        else:
            suffix = ""

        label = base + suffix
        fig.add_trace(go.Scattergl(
            x=[c_proj[i,0]], y=[c_proj[i,1]],
            mode='markers',
            marker=dict(symbol=marker_syms[i], size=12, color='darkblue'),
            name=label,
            hoverinfo='skip'
        ))

    # query author
    fig.add_trace(go.Scattergl(
        x=[q_proj[0]], y=[q_proj[1]],
        mode='markers',
        marker=dict(symbol='star', size=14, color='red'),
        name='Mystery author',
        hoverinfo='skip'
    ))

    # ── Arrowed annotations for mystery + candidates ──────────────────────────
    # Mystery author (red star)
    fig.add_annotation(
        x=q_proj[0], y=q_proj[1],
        xref='x', yref='y',
        text="Mystery",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        ax=40,   # tail offset in pixels: moves the label 40px to the right
        ay=-40,  # moves the label 40px up
        font=dict(color='red', size=12)
    )

    # Candidate authors (dark blue ◆)
    offsets = [(-40, -30), (40, -30), (0, 40)]  # [(ax,ay) for Cand1, Cand2, Cand3]
    for i in range(3):
        # build the right label
        if i == pred_idx and i == gt_idx:
            label = f"Candidate {i+1} (Predicted & Ground Truth)"
        elif i == pred_idx:
            label = f"Candidate {i+1} (Predicted)"
        elif i == gt_idx:
            label = f"Candidate {i+1} (Ground Truth)"
        else:
            label = f"Candidate {i+1}"

        fig.add_annotation(
            x=c_proj[i,0], y=c_proj[i,1],
            xref='x', yref='y',
            text= label,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            ax=offsets[i][0],
            ay=offsets[i][1],
            font=dict(color='darkblue', size=12)
        )

    # Compute precomputed regions
    bg_proj_for_regions = proj[4:]  # Background projections
    bg_ids_for_regions = bg_ids[4:]  # Background IDs
    
    # Compute precomputed regions
    mystery_id = task_authors_df['authorID'].iloc[0]  # Mystery author ID
    candidate_ids = task_authors_df['authorID'].iloc[1:4].tolist()  # 3 candidate IDs

    precomputed_regions = compute_precomputed_regions(
        bg_proj_for_regions, bg_ids_for_regions, q_proj, c_proj, pred_idx, model_name
    )
    
    # Create choices for radio buttons
    pc=json.loads(precomputed_regions)
    region_choices = ["None"] + list(pc.keys())

    print('Done processing....')
    
    return (
      fig,
    #   update(choices=display_clusters, value=display_clusters[cluster_label_query]),
      None, 
      bg_proj,  # Return background points
      bg_ids,    # Return background labels
      background_authors_embeddings_df,  # Return the DataFrame for zoom handling
      precomputed_regions,  # Return region choices
      gr.update(choices=region_choices, value="None")

    )
    # return fig, update(choices=feature_list, value=feature_list[0]),feature_list

def trigger_precomputed_region(region_name, precomputed_regions):
    """
    Simulate a zoom event for a precomputed region.
    Returns the JSON payload that would be sent to axis_ranges.
    """
    print(f"[INFO] Triggering precomputed region: {region_name}")
    print(f"precomputed_regions type: {type(precomputed_regions)}")
    # print(f"precomputed_regions content: {precomputed_regions}")
    try:
        # Parse the JSON string back to dictionary
        # precomputed_regions = json.loads(precomputed_regions) if precomputed_regions else {}
        print(f"Available regions: {len(list(precomputed_regions.keys()))}")
        # print(f"Available regions: {list(precomputed_regions.keys())}")
        if region_name == "None" or region_name not in precomputed_regions:
            return ""
        
        region = precomputed_regions[region_name]
        payload = region['bbox']
        json_payload = {
            'xaxis': [float(payload['xaxis'][0]), float(payload['xaxis'][1])],
            'yaxis': [float(payload['yaxis'][0]), float(payload['yaxis'][1])]
        }

        # js_code = trigger_plot_zoom_js(region_name, precomputed_regions)
        return json.dumps(json_payload)#, js_code
    except Exception as e:
        print(f"[ERROR] Failed to trigger precomputed region: {e}")
        return ""