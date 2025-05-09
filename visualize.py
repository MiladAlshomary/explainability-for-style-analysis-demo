import gradio as gr
import json
import numpy as np
from sklearn.manifold import TSNE
import pickle as pkl
import os
import hashlib
import pandas as pd
# from matplotlib import pyplot as plt
# import matplotlib.cm as cm
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from gradio import update

CACHE_DIR = "datasets/feature_spans_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

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

    # Mystery author box
    mystery_html = f"""
    <div style="border:1px solid #ccc; padding:10px; margin-bottom:10px;">
      <h3>Mystery Author</h3>
      <p>{data['Q_fullText']}</p>
    </div>
    """

    # Candidate boxes
    candidate_htmls = []
    for i in range(3):
        text = data[f'a{i}_fullText']
        title = f"Candidate {i+1}"
        extra_style = ""
        if data.get('rank_1') == i:
            extra_style = "transform: scale(1.05); border: 2px solid #333; padding:10px;"
        candidate_htmls.append(f"""
        <div style="border:1px solid #ccc; padding:10px; {extra_style}">
          <h4>{title}</h4>
          <p>{text}</p>
        </div>
        """)

    return mystery_html, candidate_htmls[0], candidate_htmls[1], candidate_htmls[2]

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

    # Take top features
    dimension_to_style = {dim[0]: dim[1][:cfg['top_k']] for dim in dimension_to_style.items()}


    return {
        'dimension_to_latent': dimension_to_latent, 
        'dimension_to_style' : dimension_to_style, 
        'author_embedding' : author_embedding, 
        'author_labels' : author_labels
    }

def visualize_clusters_plotly(iid, cfg, instances):
    # 1) compute all projections exactly as before

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
    cluster_label = cent_lbl[idx]
    # features of the nearest centroid to display
    feature_list = style_names[cluster_label]

    # prepare colorscale
    n_cent = len(cent_lbl)
    cent_colors = sample_colorscale("algae", [i/(n_cent-1) for i in range(n_cent)])
    # map each cluster label to its color
    color_map = { label: cent_colors[i] for i, label in enumerate(cent_lbl) }

    # background author colors pulled from their cluster label
    bg_colors = [ color_map[label] for label in bg_lbls ]

    # 2) build Plotly figure
    fig = go.Figure()
    #use layout=go.Layout(width=900, height=450) to change size.

    # background authors (light grey dots)
    fig.add_trace(go.Scattergl(
        x=bg_proj[:,0], y=bg_proj[:,1],
        mode='markers',
        marker=dict(size=6, color=bg_colors),
        name='Background authors',
        hoverinfo='skip'
    ))

    # centroids (rainbow colors + hovertext of your top-k features)
    hover_texts = [
        f"Cluster {lbl}<br>" + "<br>".join(style_names[lbl])
        for lbl in cent_lbl
    ]
    fig.add_trace(go.Scattergl(
        x=cent_proj[:,0], y=cent_proj[:,1],
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color=cent_colors),
        name='Cluster centroids',
        hovertext=hover_texts,
        hoverinfo='text'
    ))

    # query author
    fig.add_trace(go.Scattergl(
        x=[q_proj[0]], y=[q_proj[1]],
        mode='markers',
        marker=dict(symbol='star', size=14, color='black'),
        name='Mystery author',
        hoverinfo='skip'
    ))

    # three candidates
    marker_syms = ['diamond','pentagon','x']
    for i in range(3):
        label = f"Candidate {i+1}" + (" (pred)" if i==pred_idx else "")
        fig.add_trace(go.Scattergl(
            x=[c_proj[i,0]], y=[c_proj[i,1]],
            mode='markers',
            marker=dict(symbol=marker_syms[i], size=12, color='darkblue'),
            name=label,
            hoverinfo='skip'
        ))

    # layout tweaks
    fig.update_layout(
        title="t-SNE of Author Embeddings",
        width=900, height=600,
        dragmode='pan',
        legend=dict(itemsizing='constant'),
        margin=dict(l=40,r=40,t=60,b=40)
    )
    # returning the figure, the radio button choices and the complete feature list
    return fig, update(choices=feature_list, value=feature_list[0]),feature_list

def generate_feature_spans(client, text: str, features: list[str]) -> str:
    """
    Call to OpenAI to extract spans. Returns a JSON string.
    """
    prompt = f"""You are a linguistic specialist. Given a writing sample and a list of descriptive features, identify the exact text spans that demonstrate each feature.
    
    Important:
    - The headers like "Document 1:" etc are NOT part of the original text — ignore them.
    - For each feature, even if there is no match, return an empty list.
    - Only return exact phrases from the text.

    Respond in JSON format like:
    {{
      "feature1": ["span1", "span2"],
      "feature2": [],
      …
    }}

    Text:
    \"\"\"{text}\"\"\"

    Style Features:
    {features}
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

def generate_feature_spans_cached(client, instance_id: str, text: str, features: list[str]) -> dict:
    """
    Computes a cache key from instance_id + text + feature list,
    then either loads or calls the API and saves to disk.
    Returns the parsed JSON dict mapping feature->list[spans].
    """
    key = hashlib.md5((instance_id + text + "|".join(features)).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_path):
        return json.load(open(cache_path))
    else:
        raw = generate_feature_spans(client, text, features)
        mapping = json.loads(raw)
        with open(cache_path, "w") as f:
            json.dump(mapping, f, indent=2)
        return mapping

def show_spans(client, iid, selected_feature, features_list, instances, cfg):
    """
    1) Compute the full feature list
    2) Load or call the span generator
    3) Highlight the spans in the mystery text
    """
    iid = int(iid)
    inst = instances[iid]
    text = inst['Q_fullText']
    # all features used to generate the spans:
    all_feats = features_list #closest_cluster_features(instance_id, cfg, instances)
    spans_map = generate_feature_spans_cached(client, str(iid), text, all_feats)
    spans = spans_map.get(selected_feature, [])

    # naive highlight: wrap each span in <mark>
    highlighted = text
    for span in spans:
        highlighted = highlighted.replace(span, f"<mark>{span}</mark>")

    return f"""
    <div style="border:1px solid #ccc; padding:10px; margin-top:10px;">
      <h3>Highlighted Mystery Author Text</h3>
      <p>{highlighted}</p>
    </div>
    """  

