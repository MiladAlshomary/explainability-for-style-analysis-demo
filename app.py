import gradio as gr
import json

from utils.visualizations import *
from utils.llm_feat_utils import *
from utils.gram2vec_feat_utils import *
from utils.ui import *

import yaml

from dotenv import load_dotenv  
from openai import OpenAI

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
cfg = load_config()


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def visualize_and_get_g2v(iid, cfg, instances):
    plot_obj, cluster_feats, cluster_state = visualize_clusters_plotly(iid, cfg, instances)
    #compute top-10 gram2vec for this mystery text
    top_g2v_rb, top_g2v_list = get_top_gram2vec_features(iid, instances, top_n=10)
    return plot_obj, cluster_feats, cluster_state, top_g2v_rb, top_g2v_list

def app(share=False):
    instances, instance_ids = get_instances(cfg['instances_to_explain_path'])

    with gr.Blocks(title="Author Attribution Explainability Tool") as demo:
        # ── Global CSS for bolder labels & borders ──
        # gr.HTML("""
        # <style>
        #   /* Make all labels bold */
        #   .gradio-container label { font-weight: 600; }
        #   /* Make HTML boxes stand out */
        #   .gradio-container .output-html { 
        #     border: 2px solid #888; 
        #     border-radius: 4px; 
        #     padding: 0.5em; 
        #     margin-bottom: 1em;
        #   }
        # </style>
        # """)

        # ── Main Title + Short Description ──
        # gr.HTML(styled_block("<h1 style='text-align:center'>Author Attribution Explainability Tool</h1>"))

        # gr.Markdown(
        #         "This demo helps you **see inside** a deep AA model’s latent style space:  \n"
        #         "- **Cluster** your mystery document among known authors  \n"
        #         "- **Generate** human-readable style features via LLMs  \n"
        #         "- **Compare** against Gram2Vec stylometrics  \n"
        # )
        # ── Big Centered Title ──────────────────────────────────────────
        gr.HTML(styled_block("""
        <h1 style="
            text-align:center;
            font-size:3em;      /* About 48px */
            margin-bottom:0.3em;
            font-weight:700;
        ">
            Author Attribution Explainability Tool
        </h1>
        """))

        # ── Larger Description ───────────────────────────────────────────
        # gr.HTML(styled_block("""
        # <div style="
        #     font-size:1.25em;   /* About 20px */
        #     line-height:1.6;
        #     text-align:center;
        #     max-width:800px;
        #     margin:0 auto 1em auto;
        # ">
        #     This demo helps you <strong>see inside</strong> a deep AA model’s latent style space:
        #     <ul style="list-style:disc; display:inline-block; text-align:left; margin-top:0.5em;">
        #     <li><strong>Cluster</strong> your mystery document among known authors</li>
        #     <li><strong>Generate</strong> human-readable style features via LLMs</li>
        #     <li><strong>Compare</strong> against Gram2Vec stylometrics</li>
        #     </ul>
        # </div>
        # """))
        gr.HTML(styled_block("""
        <div style="
            text-align:center;
            margin: 1em auto 2em auto;
            max-width:900px;
        ">
            <p style="font-size:1.3em; line-height:1.4;">
            This demo helps you <strong>see inside</strong> a deep AA model’s latent style space.
            </p>
            <div style="
            display:flex;
            justify-content:center;
            gap:3em;
            margin-top:1em;
            ">
            <!-- CLUSTER -->
            <div style="max-width:200px;">
                <div style="font-size:2em;">🔍</div>
                <h4 style="margin:0.2em 0;">Cluster</h4>
                <p style="margin:0; font-size:1em; line-height:1.3;">
                Place your mystery text among known authors.
                </p>
            </div>
            <!-- GENERATE -->
            <div style="max-width:200px;">
                <div style="font-size:2em;">✏️</div>
                <h4 style="margin:0.2em 0;">Generate</h4>
                <p style="margin:0; font-size:1em; line-height:1.3;">
                Create human-readable style features via LLMs.
                </p>
            </div>
            <!-- COMPARE -->
            <div style="max-width:200px;">
                <div style="font-size:2em;">⚖️</div>
                <h4 style="margin:0.2em 0;">Compare</h4>
                <p style="margin:0; font-size:1em; line-height:1.3;">
                Contrast with Gram2Vec stylometric features.
                </p>
            </div>
            </div>
        </div>
        """))


        # ── Step-by-Step Guided Panel ──
        with gr.Accordion("📝 How to Use", open=True):
            gr.Markdown("""
                    1. **Select** a task from the dropdown  
                    2. Click **Run Visualization** to see latent clusters  
                    3. Pick an **LLM feature** to highlight in yellow  
                    4. Pick a **Gram2Vec feature** to highlight in blue  
                    5. Click **Show Combined Spans** to compare side-by-side  
                    """
            )


        # ── Dropdown and to select instance ─────────────────────────────
        dropdown = gr.Dropdown(
            choices=[f"Task {i}" for i in instance_ids],
            value=f"Task {instance_ids[0]}",
            label="Pick a task from the AA model’s predictions (a mystery text and its three candidate authors).",
            info="Choose which mystery document to explain"
        )


        # ── HTML outputs for author texts… ─────────────────────────────
        header  = gr.HTML()
        mystery = gr.HTML()
        with gr.Row():
            c0, c1, c2 = gr.HTML(), gr.HTML(), gr.HTML()

        dropdown.change(
            lambda iid: load_instance(int(iid.replace('Task ','')), instances),
            inputs=dropdown,
            outputs=[header, mystery, c0, c1, c2]
        )    

        # ── Visualization for clusters ─────────────────────────────
        gr.HTML(instruction_callout("Run visualization to see which author cluster contains the mystery document."))
        run_btn   = gr.Button("Run visualization")
        with gr.Row():
            with gr.Column(scale=3):
                plot_out   = gr.Plot(
                    label="Cluster Visualization",
                    elem_id="cluster-plot"
                )
            with gr.Column(scale=1):
                expl_html = """
                    <h4>What am I looking at?</h4>
                    <p>
                    This plot shows the mystery author (★) and three candidate authors (◆) 
                    in the AA model’s latent space.<br>
                    Grey ▲ are the cluster centroids—each represents an author’s average style. 
                    Documents near that ▲ share similar writing styles.<br>
                    Place your mystery text in this space to see which author‐cluster it falls into, 
                    then zoom in on a centroid to inspect its top style features.
                    </p>
                """
                gr.HTML(styled_html(expl_html))
        
        with gr.Row():
            # ── LLM Features Column ──────────────────────────────────
            with gr.Column(scale=1, min_width=0):
                # gr.Markdown("**Features from the cluster closest to the Mystery Author**")
                gr.HTML("""
                    <div style="
                        font-size: 1.3em;
                        font-weight: 600;
                        margin-bottom: 0.5em;
                    ">
                        Features from the cluster closest to the Mystery Author
                    </div>
                    """)
                features_rb = gr.Radio(choices=[], label="LLM-derived style features for this cluster")#, label="Features from the cluster closest to the Mystery Author", info="LLM-derived style features for this cluster")
                feature_list_state = gr.State() 

            # ── Gram2Vec Features Column ─────────────────────────────
            with gr.Column(scale=1, min_width=0):
                # gr.Markdown("**Top-10 Gram2Vec Features most likely to occur in Mystery Author**")
                gr.HTML("""
                    <div style="
                        font-size: 1.3em;
                        font-weight: 600;
                        margin-bottom: 0.5em;
                    ">
                        Top-10 Gram2Vec Features most likely to occur in Mystery Author
                    </div>
                    """)
                gram2vec_rb    = gr.Radio(choices=[], label="Most prominent Gram2Vec features in the mystery text")#, label="Top-10 Gram2Vec Features most likely to occur in Mystery Author", info="Most prominent Gram2Vec features in the mystery text")
                gram2vec_state = gr.State()

        run_btn.click(
            fn=lambda iid: visualize_and_get_g2v(
                int(iid.replace('Task ','')), cfg, instances
            ),
            inputs=[dropdown],
            outputs=[
                plot_out,         
                features_rb,      
                feature_list_state,
                gram2vec_rb,      
                gram2vec_state    
            ]
        )

        # ── Show combined feature‐span highlights ──
        gr.HTML(instruction_callout("Click \"Show Combined Spans\" to highlight the LLM (yellow) & Gram2Vec (blue) feature spans in the texts"))
        combined_btn  = gr.Button("Show Combined Spans")
        combined_html = gr.HTML()

        combined_btn.click(
            fn=lambda iid, sel_feat_llm, all_feats, sel_feat_g2v: show_combined_spans_all(
                client, iid.replace('Task ', ''), sel_feat_llm, all_feats, instances, sel_feat_g2v
            ),
            inputs=[dropdown, features_rb, feature_list_state, gram2vec_rb],
            outputs=[combined_html]
        )

    demo.launch(share=share)

if __name__ == "__main__":
    app(share=True)
