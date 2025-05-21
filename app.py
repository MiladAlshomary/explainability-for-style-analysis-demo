import gradio as gr
import json

from utils.visualizations import *
from utils.llm_feat_utils import *
from utils.gram2vec_feat_utils import *

import yaml

from dotenv import load_dotenv  
from openai import OpenAI

def load_config(path="config/config.yaml"):
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
            choices=['Task {}'.format(x) for x in instance_ids],
            value=str(instance_ids[0]),
            label="Select one of the tasks that was predicted by the AA model"
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
        run_btn   = gr.Button("Run Visualization")
        # trying to override default plotly legend click behavior
        # by adding a custom click handler to the legend items
        # Doesn't work yet, but the idea is to add a click handler
        gr.HTML("""
        <script>
        function handleLegendClick() {
            const plotDiv = document.querySelector('[data-cy="cluster-plot"]');
            const observer = new MutationObserver((mutations) => {
                const graphDiv = plotDiv?.querySelector('.plotly-graph-div');
                if (graphDiv) {
                    graphDiv.on('plotly_legendclick', function(eventData) {
                        const trace = eventData.fullData[eventData.curveNumber];
                        const x = trace.x;
                        const y = trace.y;
                        
                        // Handle single-point traces (candidates/query)
                        let xRange, yRange;
                        if (x.length === 1 && y.length === 1) {
                            xRange = [x[0]-0.5, x[0]+0.5];
                            yRange = [y[0]-0.5, y[0]+0.5];
                        } else {
                            // Multi-point traces (centroids/background)
                            const xMin = Math.min(...x);
                            const xMax = Math.max(...x);
                            const yMin = Math.min(...y);
                            const yMax = Math.max(...y);
                            const xPadding = (xMax - xMin) * 0.1;
                            const yPadding = (yMax - yMin) * 0.1;
                            xRange = [xMin - xPadding, xMax + xPadding];
                            yRange = [yMin - yPadding, yMax + yPadding];
                        }
                        
                        Plotly.relayout(graphDiv, {
                            'xaxis.range': xRange,
                            'yaxis.range': yRange
                        });
                        
                        return false; // Prevent default hide/show behavior
                    });
                    observer.disconnect();
                }
            });
            observer.observe(plotDiv, { childList: true, subtree: true });
        }
        
        // Initial setup
        document.addEventListener("DOMContentLoaded", handleLegendClick);
        // Reconnect when plot updates
        document.addEventListener("DOMNodeInserted", handleLegendClick);
        </script>
        """)
        
        plot_out   = gr.Plot(
            label="Cluster Visualization",
            elem_id="cluster-plot"
        )
        features_rb = gr.Radio(choices=[], label="Closest Cluster Features")
        feature_list_state = gr.State() # placeholder for your extra output, invisible on the UI

        run_btn.click(
            fn=lambda iid: visualize_clusters_plotly(
                int(iid.replace('Task ','')), cfg, instances
            ),
            inputs=[dropdown],
            outputs=[plot_out, features_rb, feature_list_state]
        )

        # Use static config-driven feature list
        gram2vec_feature_list = [
            "pos_unigrams:NOUN", "pos_unigrams:VERB", "pos_unigrams:DET",
            "pos_bigrams:ADJ_NOUN", "func_words:the", "func_words:of", "punctuation:.",
            "letters:uppercase", "emojis:😂", "dep_labels:nsubj", "morph_tags:Number=Plur",
            "sentences:declarative"
        ]

        gram2vec_rb = gr.Radio(choices=gram2vec_feature_list, label="Gram2Vec Features")

        # Show combined feature‐span highlighting
        combined_btn  = gr.Button("Show Combined Spans")
        combined_html = gr.HTML()#"""
        #         <style>
        #         .mark-llm     { background-color: #fff176; }   /* yellow-200 */
        #         .mark-gram    { background-color: #90caf9; }   /* light-blue-300 */
        #         </style>
        #         """
        # )

        gr.Markdown("### Highlight LLM (yellow) & Gram2Vec (blue) Spans Together")

        combined_btn.click(
            fn=lambda iid, sel_feat_llm, all_feats, sel_feat_g2v: show_combined_spans_all(
                client, iid.replace('Task ', ''), sel_feat_llm, all_feats, instances, sel_feat_g2v
            ),
            inputs=[dropdown, features_rb, feature_list_state, gram2vec_rb],
            outputs=[combined_html]
        )

        # ── LLM Feature Highlighting ─────────────────────────────
        # show_btn = gr.Button("Show LLM Feature Spans")
        # highlighted_out = gr.HTML()

        # show_btn.click(fn=lambda iid, sel_feat, all_feats: show_both_spans(client, iid.replace('Task ',''), sel_feat, all_feats, instances, cfg),
        #                inputs=[dropdown, features_rb, feature_list_state],
        #                outputs=[highlighted_out])

        # # ── Gram2Vec Feature Highlighting ─────────────────────────────
        # gr.Markdown("### Compare with Gram2Vec Feature Spans")

        
        # gram_btn = gr.Button("Show Gram2Vec Feature Spans")
        # gram_html = gr.HTML()

        # gram_btn.click(
        #     fn=lambda iid, sel_feat: show_gram2vec_spans_all(
        #         int(iid.replace('Task ','')), sel_feat, instances
        #     ),
        #     inputs=[dropdown, gram2vec_rb],
        #     outputs=[gram_html]
        # )


    demo.launch(share=share)

if __name__ == "__main__":
    app(share=True)
