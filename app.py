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


# ── load once at startup ────────────────────────────────────────
GRAM2VEC_SHORTHAND = load_code_map()  

def app(share=False):
    instances, instance_ids = get_instances(cfg['instances_to_explain_path'])

    with gr.Blocks(title="Author Attribution Explainability Tool") as demo:
        # ── Big Centered Title ──────────────────────────────────────────
        gr.HTML(styled_block("""
        <h1 style="
            text-align:center;
            font-size:3em;      /* About 48px */
            margin-bottom:0.3em;
            font-weight:700;
        ">
            Author Attribution (AA) Explainability Tool
        </h1>
        """))

        gr.HTML(styled_block("""
        <div style="
            text-align:center;
            margin: 1em auto 2em auto;
            max-width:900px;
        ">
            <p style="font-size:1.3em; line-height:1.4;">
            This demo helps you <strong>see inside</strong> a deep AA model’s latent style space.
            </p>
            <p style="font-size:0.9em; line-height:1.4;">
            Currently you are inspecting <a href="https://huggingface.co/rrivera1849/LUAR-MUD">LUAR</a> with pre-defined AA tasks from the <a href="https://www.iarpa.gov/images/research-programs/HIATUS/IARPA_HIATUS_Phase_1_HRS_Data.to_DAC_20240610.pdf">HRS dataset </a> 
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
                Place your AA task with respect to other background authors.
                </p>
            </div>
            <!-- GENERATE -->
            <div style="max-width:200px;">
                <div style="font-size:2em;">✏️</div>
                <h4 style="margin:0.2em 0;">Generate</h4>
                <p style="margin:0; font-size:1em; line-height:1.3;">
                Describe your investigated authors' writing style via human-readable LLM-generated style features.
                </p>
            </div>
            <!-- COMPARE -->
            <div style="max-width:200px;">
                <div style="font-size:2em;">⚖️</div>
                <h4 style="margin:0.2em 0;">Compare</h4>
                <p style="margin:0; font-size:1em; line-height:1.3;">
                Contrast with <a href=""https://github.com/eric-sclafani/gram2vec>Gram2Vec</a> stylometric features.
                </p>
            </div>
            </div>
        </div>
        """))


        # ── Step-by-Step Guided Panel ──
        with gr.Accordion("📝 How to Use", open=True):
            gr.Markdown("""
                    1. **Select** a pre-defined task from the dropdown
                    2. Click **Run Visualization** to see where the authors are located in the AA model's space
                    3. Pick an **LLM feature** to highlight in yellow  
                    4. Pick a **Gram2Vec feature** to highlight in blue  
                    5. Click **Show Combined Spans** to compare side-by-side
                    """
            )


        # ── Dropdown and to select instance ─────────────────────────────
        # Load default instance values for display
        default_outputs = load_instance(0, instances)
        gr.HTML("""
                    <div style="
                        font-size: 1.3em;
                        font-weight: 600;
                        margin-bottom: 0.5em;
                    ">
                        Pick a pre-defined task to investigate (a mystery text and its three candidate authors)
                    </div>
                    """)

        task_dropdown = gr.Dropdown(
            choices=[f"Task {i}" for i in instance_ids],
            value=f"Task {instance_ids[0]}",
            label="Choose which mystery document to explain",
        )


        # ── HTML outputs for author texts… ─────────────────────────────
        header  = gr.HTML(value=default_outputs[0])
        mystery = gr.HTML(value=default_outputs[1])
        with gr.Row():
            c0, c1, c2 = gr.HTML(value=default_outputs[2]), gr.HTML(value=default_outputs[3]), gr.HTML(value=default_outputs[4])

        task_dropdown.change(
            lambda iid: load_instance(int(iid.replace('Task ','')), instances),
            inputs=task_dropdown,
            outputs=[header, mystery, c0, c1, c2]
        )    

        # ── Visualization for clusters ─────────────────────────────
        gr.HTML(instruction_callout("Run visualization to see which author cluster contains the mystery document."))
        run_btn   = gr.Button("Run visualization")
        bg_proj_state = gr.State()
        bg_lbls_state = gr.State()
        bg_authors_df = gr.State()  # Holds the background authors DataFrame
        with gr.Row():
            with gr.Column(scale=3):
                # plot_out   = gr.Plot(
                #     label="Cluster Visualization",
                #     elem_id="cluster-plot"
                # )
                axis_ranges = gr.Textbox(visible=False, elem_id="axis-ranges")
                plot = gr.Plot(
                    label="Cluster Visualization",
                    elem_id="cluster-plot",
                )
                plot.change(
                    fn=None,
                    inputs=[plot],
                    outputs=[axis_ranges],
                    js="""
                    function(){
                        console.log("------------>[JS] plot.change() triggered<------------");

                        let attempts = 0;
                        const maxAttempts = 50;

                        const tryAttach = () => {
                            const gd = document.querySelector('#cluster-plot .js-plotly-plot');
                            if (!gd) {
                                if (++attempts < maxAttempts) {
                                    requestAnimationFrame(tryAttach);
                                } else {
                                    console.error(" ------------>Could not find .js-plotly-plot after multiple attempts.<------------");
                                }
                                return;
                            }

                            if (gd.__zoomListenerAttached) {
                                console.log("------------>Zoom listener already attached.<------------");
                                return;
                            }

                            gd.__zoomListenerAttached = true;
                            console.log("------------>Zoom listener attached!<------------");

                            gd.on('plotly_relayout', (ev) => {
                                if (
                                    ev['xaxis.range[0]'] === undefined ||
                                    ev['xaxis.range[1]'] === undefined ||
                                    ev['yaxis.range[0]'] === undefined ||
                                    ev['yaxis.range[1]'] === undefined
                                ) return;

                                const payload = {
                                    xaxis: [ev['xaxis.range[0]'], ev['xaxis.range[1]']],
                                    yaxis: [ev['yaxis.range[0]'], ev['yaxis.range[1]']]
                                };

                                const txtbox = document.querySelector('#axis-ranges textarea');
                                if (txtbox) {
                                    txtbox.value = JSON.stringify(payload);
                                    txtbox.dispatchEvent(new Event('input', { bubbles: true }));
                                    console.log("------------> Zoom payload dispatched:<------------", payload);
                                } else {
                                    console.warn("------------> No hidden textbox found to write zoom payload.<------------");
                                }
                            });
                        };

                        requestAnimationFrame(tryAttach);
                        return '';
                    }
                    """
                )


            with gr.Column(scale=1):
                expl_html = """
                    <h4>What am I looking at?</h4>
                    <p>
                    This plot shows the mystery author (★) and three candidate authors (◆) 
                    in the AA model’s latent space.<br>
                    Grey ▲ are identified salient regions in the AA model's space—each has a specific writing style. 
                    Hover over the ▲ to see the top 10 writing style features<br>
                    Place your mystery text in this space to see which author‐cluster it falls into, 
                    then zoom in on a centroid to inspect its top style features.
                    </p>
                """
                gr.HTML(styled_html(expl_html))
        
        # Add handler for filtered points
        filtered_points = gr.Textbox(label="Filtered Points")  # Hidden component for filtered points
        axis_ranges.change(
            fn=handle_zoom, 
            inputs=[axis_ranges, bg_proj_state, bg_lbls_state, bg_authors_df], 
            outputs=[filtered_points]
        )

        # ── Dynamic Cluster Choice dropdown ──────────────────────────────────
        gr.HTML(instruction_callout("Choose a cluster from the dropdown below to inspect whether its features appear in the mystery author’s text."))
        cluster_dropdown = gr.Dropdown(choices=[], label="Select Cluster to Inspect")
        style_map_state = gr.State()  # Holds the mapping of cluster->features
        
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
                        LLM-derived style  features prominent in the selected cluster
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
                        Gram2Vec Features prominent in the selected cluster
                    </div>
                    """)
                gram2vec_rb    = gr.Radio(choices=[], label="Gram2Vec features for this cluster")#, label="Top-10 Gram2Vec Features most likely to occur in Mystery Author", info="Most prominent Gram2Vec features in the mystery text")
                gram2vec_state = gr.State()

        # ── Visualization button click ───────────────────────────────
        run_btn.click(
            fn=lambda iid: visualize_clusters_plotly(
                int(iid.replace('Task ','')), cfg, instances
            ),
            inputs=[task_dropdown],
            outputs=[plot, cluster_dropdown, style_map_state, bg_proj_state, bg_lbls_state, bg_authors_df]
        )

        # When a cluster is selected, split features and populate radio buttons
        def on_cluster_change(selected_cluster, style_map):
            cluster_key = extract_cluster_key(selected_cluster)
            all_feats = style_map[cluster_key]
            llm_feats, g2v_feats = split_features(all_feats)
            # print(f"Selected cluster: {selected_cluster} ({cluster_key})")
            # print(f"LLM features: {llm_feats}")

            # Add "None" as a default selectable option
            llm_feats = ["None"] + llm_feats

            # filter out any g2v feature without a shorthand
            filtered_g2v = []
            for feat in g2v_feats:
                if get_shorthand(feat) is None:
                    print(f"Skipping Gram2Vec feature without shorthand: {feat}")
                else:
                    filtered_g2v.append(feat)
            
            # Add "None" as a default selectable option
            filtered_g2v = ["None"] + filtered_g2v

            return (
                gr.update(choices=llm_feats, value=llm_feats[0]),
                gr.update(choices=filtered_g2v, value=filtered_g2v[0]),
                llm_feats
            )

        cluster_dropdown.change(
            fn=on_cluster_change,
            inputs=[cluster_dropdown, style_map_state],
            outputs=[features_rb, gram2vec_rb , feature_list_state] #adding feature_list_state to persisit all llm features in the app state
        )


        # ── Show combined feature‐span highlights ──
        # combined callout + legend in one HTML block
        gr.HTML(
            instruction_callout(
                "Click \"Show Combined Spans\" to highlight the LLM (yellow) & Gram2Vec (blue) feature spans in the texts"
            )
            + """
            <div style="
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 2em;
                margin-top: 0.5em;
                font-size: 0.9em;
            ">
            <div style="display: flex; align-items: center; gap: 0.5em; font-weight: 600; font-size: 1.5em;">
                <span style="
                    display: inline-block;
                    width: 1.5em; height: 1.5em;
                    background: #FFEB3B;      /* bright yellow */
                    border: 1px solid #666;
                    vertical-align: middle;
                "></span>
                LLM feature
            </div>
            <div style="display: flex; align-items: center; gap: 0.5em; font-weight: 600; font-size: 1.5em;">
                <span style="
                    display: inline-block;
                    width: 1.5em; height: 1.5em;
                    background: #5CB3FF;      /* clearer blue */
                    border: 1px solid #666;
                    vertical-align: middle;
                "></span>
                Gram2Vec feature
            </div>
            </div>
            """
        )


        combined_btn  = gr.Button("Show Combined Spans")
        combined_html = gr.HTML()

        # print(f"in app: all_feats={feature_list_state.value}")
        # print(f"in app: sel_feat_llm={features_rb.value}")


        combined_btn.click(
            fn=lambda iid, sel_feat_llm, all_feats, sel_feat_g2v: show_combined_spans_all(
                client, iid.replace('Task ', ''), sel_feat_llm, all_feats, instances, sel_feat_g2v
            ),
            inputs=[task_dropdown, features_rb, feature_list_state, gram2vec_rb],
            outputs=[combined_html]
        )
        # mapping -->
        # iid = task_dropdown.value
        # sel_feat_llm = features_rb.value
        # all_feats = feature_list_state.value
        # sel_feat_g2v = gram2vec_rb.value

    demo.launch(share=share)

if __name__ == "__main__":
    app(share=True)
