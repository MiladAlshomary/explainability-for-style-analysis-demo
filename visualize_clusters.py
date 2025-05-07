import pandas as pd
import json
import sklearn
import glob
import pickle as pkl
from sklearn.model_selection import train_test_split
from collections import Counter
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
import numpy as np
from sklearn.manifold import TSNE
from nltk import sent_tokenize
import sys
import matplotlib.cm as cm

import IPython.display as display
from IPython.display import display, Javascript, HTML
import ipywidgets as widgets

def create_style_features_box(feats, widgets, output):
    # Create a checkbox for each item
    checkboxes = [widgets.Checkbox(value=False, description=item) for item in feats]
    
    # Create a function to read selected values
    def annotate_selected_features(_):
        selected = [cb.description for cb in checkboxes if cb.value]
        msg = "You selected: " + ", ".join(selected) if selected else "Nothing selected."
        
        # Trigger JavaScript alert
        with output:
            output.clear_output()
            display(HTML(f"""
                <script type="text/javascript">
                    alert("{msg}");
                </script>
            """))
        
    # Button to trigger reading the selected checkboxes
    submit_button = widgets.Button(description='annotate features in texts')
    submit_button.on_click(annotate_selected_features)
    
    # Display checkboxes and the button
    box = widgets.VBox(checkboxes + [submit_button])
    display(box)

# Helper functions
def update_annot(annot, sc, ind, names):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "Cluster(s): {}\nTop K Features:\n{}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)
    

def mouse_click(fig, sc, ax, annot, names, event):
    vis = annot.get_visible()
    if vis:
        return

    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(annot, sc, ind, names)
            annot.set_visible(True)
            fig.canvas.draw_idle()

def key_release(fig, sc, ax, annot, names, event):
    vis = annot.get_visible()
    if event.key == 'escape':
        annot.set_visible(False)
        fig.canvas.draw_idle()

def load_interp_space(interp_folder_path, style_feat_clm, top_k, only_llm_feats, only_gram2vec_feats):
    interp_space_path      = interp_folder_path + 'interpretable_space.pkl'
    interp_space_rep_path  = interp_folder_path + 'interpretable_space_representations.json'
    gram2vec_feats_path    = interp_folder_path + '/../gram2vec_feats.csv'
    clustered_authors_path = interp_folder_path + 'train_authors.pkl'

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
    dimension_to_style  = {x[0]: [feat[0] for feat in sorted(x[1].items(), key=lambda feat_w:-feat_w[1])] for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_clm].tolist())}

    if only_llm_feats:
        #print('only llm feats')
        dimension_to_style = {dim[0]:[feat for feat in dim[1] if feat not in gram2vec_feats] for dim in dimension_to_style.items()}

    if only_gram2vec_feats:
        #print('only gra2vec feats')
        dimension_to_style = {dim[0]:[feat for feat in dim[1] if feat in gram2vec_feats] for dim in dimension_to_style.items()}

    # Take top features
    dimension_to_style = {dim[0]: dim[1][:top_k] for dim in dimension_to_style.items()}


    return {
        'dimension_to_latent': dimension_to_latent, 
        'dimension_to_style' : dimension_to_style, 
        'author_embedding' : author_embedding, 
        'author_labels' : author_labels
    }

class ExplainabilityDemo():

    def __init__(self):
        interp_space_path    = './datasets/luar_clusters_07/'
        instances_to_explain_path = './datasets/hrs_explanations.json'
        
        self.style_feat_clm = 'llm_tfidf_weights'
        self.top_k=10
        self.only_llm_feats=True
        self.only_gram2vec_feats=False
        
        self.instances_to_explain = json.load(open(instances_to_explain_path))
        self.interp_space = load_interp_space(interp_space_path, self.style_feat_clm, self.top_k, self.only_llm_feats, self.only_gram2vec_feats)

    def visualize_clusters(self, instance_id, out):

        # print('\n', 'Model Prediction:',  self.instances_to_explain[instance_id]['latent_rank'], '\n',
        #       'Interp Prediction:', self.instances_to_explain[instance_id]['interp_rank'], '\n',
        #       'Selected Clusters:', self.instances_to_explain[instance_id]['rep_clusters'], '\n'
        # )
        
        dimension_to_latent =    self.interp_space['dimension_to_latent']
        dimension_to_style  =    self.interp_space['dimension_to_style']
        
        background_author_embedding    = np.array(self.interp_space['author_embedding'])
        background_author_labels       =    self.interp_space['author_labels']

        query_author_latent   = np.array(self.instances_to_explain[instance_id]['author_latents'][:1])
        candid_author_latents = np.array(self.instances_to_explain[instance_id]['author_latents'][1:])
        predicted_author_idx = self.instances_to_explain[instance_id]['latent_rank'][0]
    
        selected_clusters = self.instances_to_explain[instance_id]['rep_clusters']
    
        centroid_embeddings = np.array([x[1] for x in dimension_to_latent.items() ])
        centroid_labels = np.array([x[0] for x in dimension_to_latent.items() ])
        
        #centroid_embeddings = np.array([x[1] for x in dimension_to_latent.items() if x[0] in selected_clusters])
        #centroid_labels = np.array([x[0] for x in dimension_to_latent.items() if x[0] in selected_clusters])
    
        # background_author_embedding = [ba for ba, label in zip(background_author_embedding, background_author_labels) if label in selected_clusters]
        # background_author_labels    = [label for label in background_author_labels if label in selected_clusters]
    
        num_background_authors = len(background_author_labels)
        num_centroid = len(centroid_labels)
    
        all_embeddings = np.concatenate((query_author_latent, candid_author_latents, background_author_embedding, centroid_embeddings))
    
        all_proj_embeddings = TSNE(n_components=2, learning_rate='auto',
                          init='random', perplexity=3).fit_transform(all_embeddings)
    
        
        query_author_proj   = all_proj_embeddings[0]
        candid_authors_proj = all_proj_embeddings[1:4]
        background_author_proj = all_proj_embeddings[4:4+num_background_authors]
        centroid_proj       = all_proj_embeddings[4+num_background_authors:]
    
    
        annotation_names = np.array(["\n".join(["{}. {}".format(i, f) for i, f in enumerate(dimension_to_style[item[0]])]) for item in dimension_to_latent.items()])
        #names = np.array(['\n'.join(sent_tokenize(dimension_to_style[item[0]])) for item in dimension_to_latent.items()])
    
    
    
        #fig,ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection="3d"))
        fig,ax = plt.subplots(figsize=(12, 8))
    
        #labels_to_show = [l for l in set(background_author_labels) if l in selected_clusters]
        labels_to_show = set(background_author_labels)
    
        # First viusalize the authors
        self.visualize_points(fig, ax, background_author_proj, background_author_labels, labels_to_show, out)
        # Visualize the centroids
        self.visualize_points(fig, ax, centroid_proj, centroid_labels, labels_to_show, out, add_annotation=True, names_dict=annotation_names)
        # Visualize the query and candidate authors
        self.visualize_query_and_candidate_authors(fig, ax, query_author_proj, candid_authors_proj, predicted_author_idx)
    
    
        plt.show()
    
    def visualize_points(self, fig, ax, points, points_labels, labels, output, add_annotation=False, names_dict=None):
        colors = iter(cm.rainbow(np.linspace(0, 1, len(labels))))
    
        if add_annotation:
    
            # # Click event handler
            # @output.capture(clear_output=True)
            # def on_click(fig, sc, ax, annot, names, event):
            #     if event.xdata is not None and event.ydata is not None:
            #         print(f"Clicked at: ({event.xdata:.2f}, {event.ydata:.2f})")
                    
            #         # Inject JavaScript call using HTML and IPython
            #         js_trigger = f"""
            #         <script>
            #             showAlert({event.xdata}, {event.ydata});
            #         </script>
            #         """
            #         #display(HTML(js_trigger))
    
    
            label_xs = points[:,0]
            label_ys = points[:,1]
            #label_zs = points[:,2]
    
            sc = plt.scatter(label_xs, label_ys, color=list(colors), alpha=0.5, marker='^')
    
    
            annot = ax.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)
            fig.canvas.mpl_connect("button_press_event", lambda x: mouse_click(fig, sc, ax, annot, names_dict, x))
            #fig.canvas.mpl_connect("button_press_event", lambda x: on_click(fig, sc, ax, annot, names_dict, x))
            fig.canvas.mpl_connect('key_release_event', lambda x: key_release(fig, sc, ax, annot, names_dict, x))
    
        
        else:
            for label in set(labels):
                label_xs = [v[0] for v in zip(points[:,0], points_labels) if v[1] == label]
                label_ys = [v[0] for v in zip(points[:,1], points_labels) if v[1] == label]
                #label_zs = [v[0] for v in zip(points[:,2], points_labels) if v[1] == label]
                next_color = next(colors)
                sc = plt.scatter(label_xs, label_ys, alpha=0.5,color=next_color)
    
    
    def visualize_query_and_candidate_authors(self, fig, ax, query_author_proj, candid_authors_proj, predicted_author_idx):
    
        predicted_author = np.array([a for i, a in enumerate(candid_authors_proj) if i == predicted_author_idx])
        other_authors = np.array([a for i, a in enumerate(candid_authors_proj) if i != predicted_author_idx])
    
        sc = plt.scatter(query_author_proj[0], query_author_proj[1],  color='gray',  marker='*', label='Query Author')
        #sc = plt.scatter(predicted_author[:,0], predicted_author[:,1],color='gray', marker='d', label='Candidate author {}(Predicted Author)'.format(predicted_author_idx))
        
        sc = plt.scatter(candid_authors_proj[0,0], candid_authors_proj[0,1], color='gray', marker='D', label='Candidate Author {} {}'.format(1,'' if predicted_author_idx != 0 else "(Predicted)"))
        sc = plt.scatter(candid_authors_proj[1,0], candid_authors_proj[1,1], color='gray', marker='P', label='Candidate Author {} {}'.format(2,'' if predicted_author_idx != 1 else "(Predicted)"))
        sc = plt.scatter(candid_authors_proj[2,0], candid_authors_proj[2,1], color='gray', marker='X', label='Candidate Author {} {}'.format(3,'' if predicted_author_idx != 2 else "(Predicted)"))

        plt.legend()
    
        # annotation_names_dict = np.array(["\n".join(["{}. {}".format(i, f) for i, f in enumerate(dimension_to_style[item[0]])]) for item in dimension_to_latent.items()])
    
        # annot = ax.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
        #                     bbox=dict(boxstyle="round", fc="w"),
        #                     arrowprops=dict(arrowstyle="->"))
        # annot.set_visible(False)
        # fig.canvas.mpl_connect("motion_notify_event", lambda x: hover(fig, sc, ax, annot, annotation_names_dict, x))