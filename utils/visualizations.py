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

def clean_text(text: str) -> str:
    """
    Cleans the text by replacing HTML tags with their escaped versions.
    """
    return text.replace('<','&lt;').replace('>','&gt;')

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
                           init='random', perplexity=3).fit_transform(embeddings)
        cache[hash_key] = tsne_result
        with open(cache_path, 'wb') as f:
            pkl.dump(cache, f)
        return tsne_result  

def load_interp_space(cfg):
    interp_space_path      = cfg['interp_space_path'] + 'interpretable_space.pkl'
    interp_space_rep_path  = cfg['interp_space_path'] + 'interpretable_space_representations.json'
    gram2vec_feats_path    = cfg['interp_space_path'] + '/../gram2vec_feats.csv'
    clustered_authors_path = cfg['interp_space_path'] + 'train_authors.pkl'

    # Load authors embeddings and their cluster labels
    clustered_authors_df = pd.read_pickle(clustered_authors_path)
    clustered_authors_df = clustered_authors_df[clustered_authors_df.cluster_label != -1]
    author_embedding = clustered_authors_df.author_embedding.tolist()
    author_labels    = clustered_authors_df.cluster_label.tolist()

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
        'author_labels' : author_labels
    }

def visualize_clusters_plotly(iid, cfg, instances):
    print("Generating cluster visualization")

    iid = int(iid)
    interp      = load_interp_space(cfg)
    dim2lat     = interp['dimension_to_latent']
    style_names = interp['dimension_to_style']
    bg_emb      = np.array(interp['author_embedding'])
    bg_lbls     = interp['author_labels']

    inst         = instances[iid]
    q_lat        = np.array(inst['author_latents'][:1])
    c_lat        = np.array(inst['author_latents'][1:])
    pred_idx     = inst['latent_rank'][0]
    gt_idx       = inst['gt_idx']

    cent_emb = np.array([v for _,v in dim2lat.items()])
    cent_lbl = np.array([k for k,_ in dim2lat.items()])

    all_emb = np.vstack([q_lat, c_lat, bg_emb, cent_emb])
    proj    = compute_tsne_with_cache(all_emb)

    # split
    q_proj    = proj[0]
    c_proj    = proj[1:4]
    bg_proj   = proj[4:4+len(bg_lbls)]
    cent_proj = proj[4+len(bg_lbls):]


    # find nearest centroid
    dists = np.linalg.norm(cent_proj - q_proj, axis=1)
    idx   = int(np.argmin(dists))
    cluster_label_query = cent_lbl[idx]
    # features of the nearest centroid to display
    feature_list = style_names[cluster_label_query]

    cluster_labels_per_candidate = [
        cent_lbl[int(np.argmin(np.linalg.norm(cent_proj - c_proj[i], axis=1)))]
        for i in range(c_proj.shape[0])
    ]

    # prepare colorscale
    n_cent = len(cent_lbl)
    cent_colors = sample_colorscale("algae", [i/(n_cent-1) for i in range(n_cent)])
    # map each cluster label to its color
    color_map = { label: cent_colors[i] for i, label in enumerate(cent_lbl) }

    # uncomment the following line to show background authors
    ## background author colors pulled from their cluster label
    # bg_colors = [ color_map[label] for label in bg_lbls ]

    # 2) build Plotly figure
    fig = go.Figure()
    
    fig.update_layout(
        template='plotly_white',
        margin=dict(l=40,r=40,t=60,b=40),
        autosize=True,
        hovermode='closest')


    # uncomment the following line to show background authors
    ## background authors (light grey dots)
    # fig.add_trace(go.Scattergl(
    #     x=bg_proj[:,0], y=bg_proj[:,1],
    #     mode='markers',
    #     marker=dict(size=6, color=bg_colors),
    #     name='Background authors',
    #     hoverinfo='skip'
    # ))

    # centroids (rainbow colors + hovertext of your top-k features)
    hover_texts = [
        f"Cluster {lbl}<br>" + "<br>".join(style_names[lbl])
        for lbl in cent_lbl
    ]
    fig.add_trace(go.Scattergl(
        x=cent_proj[:,0], y=cent_proj[:,1],
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color="#d3d3d3"),#color=cent_colors
        name='Cluster centroids',
        hovertext=hover_texts,
        hoverinfo='text'
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

    # Prepare outputs for the new cluster‐dropdown UI
    # all_clusters = sorted(style_names.keys())
    # --- build display names for the dropdown ---
    sorted_labels = sorted([int(lbl) for lbl in cent_lbl])
    display_clusters = []
    for lbl in sorted_labels:
        name = f"Cluster {lbl}"
        if lbl == cluster_label_query:
            name += " (closest to mystery author)"
        matching_indices = [i + 1 for i, val in enumerate(cluster_labels_per_candidate) if int(val) == lbl]
        if matching_indices:
            if len(matching_indices) == 1:
                name += f" (closest to Candidate {matching_indices[0]} author)"
            else:
                candidate_str = ", ".join(f"Candidate {i}" for i in matching_indices)
                name += f" (closest to {candidate_str} authors)"
        display_clusters.append(name)
    # print(f"All clusters: {all_clusters}")
    # return: figure, dropdown payload, full style_map
    return (
      fig,
      update(choices=display_clusters, value=display_clusters[cluster_label_query]),
      style_names
    )
    # return fig, update(choices=feature_list, value=feature_list[0]),feature_list


def extract_cluster_key(display_label: str) -> int:
    """
    Given a dropdown label like
      "Cluster 5 (closest to mystery author; closest to Candidate 1 author)"
    returns the integer 5.
    """
    m = re.match(r"Cluster\s+(\d+)", display_label)
    if not m:
        raise ValueError(f"Unrecognized cluster label: {display_label}")
    return int(m.group(1))
