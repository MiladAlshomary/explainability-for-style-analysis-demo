import gradio as gr
import json

from visualize import *

import yaml

from dotenv import load_dotenv  
from openai import OpenAI

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
cfg = load_config()


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def app(share=False):
    instances, instance_ids = get_instances(cfg['instances_to_explain_path'])

    with gr.Blocks(title="Author Attribution Explainability Tool") as demo:
        gr.Markdown("# Author Attribution Explainability Tool")

        # ── Dropdown and to select instance ─────────────────────────────
        dropdown = gr.Dropdown(
            choices=[str(x) for x in instance_ids],
            value=str(instance_ids[0]),
            label="Select instance"
        )

        # ── HTML outputs for author texts… ─────────────────────────────
        mystery = gr.HTML()
        with gr.Row():
            c0, c1, c2 = gr.HTML(), gr.HTML(), gr.HTML()

        dropdown.change(
            lambda iid: load_instance(iid, instances),
            inputs=dropdown,
            outputs=[mystery, c0, c1, c2]
        )    

        # ── Visualization for clusters ─────────────────────────────
        run_btn   = gr.Button("Run Visualization")
        # with gr.Row():
        plot_out   = gr.Plot()
        features_rb = gr.Radio(choices=[], label="Closest Cluster Features")
        feature_list_state = gr.State() # placeholder for your extra output, invisible on the UI

        run_btn.click(
            fn=lambda iid: visualize_clusters_plotly(
                int(iid), cfg, instances
            ),
            inputs=[dropdown],
            outputs=[plot_out, features_rb, feature_list_state]
        )

        # Show feature‐span highlighting
        show_btn       = gr.Button("Show Feature Spans")
        highlighted_out = gr.HTML()

        show_btn.click(fn=lambda iid, sel_feat, all_feats: show_spans(client, iid, sel_feat, all_feats, instances, cfg),
                       inputs=[dropdown, features_rb, feature_list_state],
                       outputs=[highlighted_out])

        # features_rb = gr.Radio(choices=features_rb, label="Closest Cluster Features")



    demo.launch(share=share)

if __name__ == "__main__":
    app(share=True)
