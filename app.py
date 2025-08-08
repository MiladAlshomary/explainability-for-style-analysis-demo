import gradio as gr
import json

from utils.visualizations import *
from utils.llm_feat_utils import *
from utils.gram2vec_feat_utils import *
from utils.interp_space_utils import *
from utils.ui import *

import os
os.environ["GRADIO_TEMP_DIR"] = "./datasets/temp"  # Set a custom temp directory for Gradio
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

import yaml
import argparse

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

def validate_ground_truth(gt1, gt2, gt3):
    selected = [gt1, gt2, gt3]
    selected_count = sum(selected)

    if selected_count > 1:
        return None, "Please select only one ground truth author."
    elif selected_count == 0:
        return None, "No ground truth author selected."

    index = selected.index(True)
    return index, f"Candidate {index+1} is marked as the ground truth author."


def app(share=False, use_cluster_feats=False):
    instances, instance_ids = get_instances(cfg['instances_to_explain_path'])

    interp      = load_interp_space(cfg)
    clustered_authors_df = interp['clustered_authors_df'][:1000]
    clustered_authors_df['fullText'] = clustered_authors_df['fullText'].map(lambda l: l[:3]) # Take at most 3 texts per author

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
            <!-- Visualize -->
            <div style="max-width:200px;">
                <div style="font-size:2em;">🔍</div>
                <h4 style="margin:0.2em 0;">Visualize</h4>
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
                    1. **Select** a model and a task source (pre-defined or custom)
                    2. Click **Load Task & Generate Embeddings** to load the task and generate embeddings
                    3. **Run Visualization** to see the mystery author and candidates in the AA model's latent space
                    4. **Zoom** into the visualization to select a cluster of background authors
                    5. Pick an **LLM feature** to highlight in yellow  
                    6. Pick a **Gram2Vec feature** to highlight in blue  
                    7. Click **Show Combined Spans** to compare side-by-side
                    """
            )

        # ── Model Selection ─────────────────────────────────
        model_radio = gr.Radio(
            choices=[
                'gabrielloiseau/LUAR-MUD-sentence-transformers',
                'gabrielloiseau/LUAR-CRUD-sentence-transformers',
                'miladalsh/light-luar',
                'AnnaWegmann/Style-Embedding',
                'Other'
            ],
            value='gabrielloiseau/LUAR-MUD-sentence-transformers',
            label='Choose a Model to inspect'
        )
        print(f"Model choices: {model_radio.choices}")
        print(f"Model default: {model_radio.value}")
        custom_model = gr.Textbox(
            label='Custom Model ID',
            placeholder='Enter your Hugging Face Model ID here',
            visible=False,
            interactive=True
        )
        # Show the textbox when 'Other' is selected
        model_radio.change(
            fn=toggle_custom_model,
            inputs=[model_radio],
            outputs=[custom_model]
        )

        # ── Task Source Selection ─────────────────────────────────
        task_mode = gr.Radio(
            choices=["Predefined HRS Task", "Upload Your Own Task"],
            value="Predefined HRS Task",
            label="Select Task Source"
        )

        ground_truth_author = gr.State()  # To store the index of the ground truth author

        with gr.Column():
            with gr.Column(visible=True) as predefined_container:
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
            with gr.Column(visible=False) as custom_container:
                gr.HTML("""
                    <div style="
                        font-size: 1.3em;
                        font-weight: 600;
                        margin-bottom: 0.5em;
                    ">
                        Upload your own task
                    </div>
                    """)
                mystery_input   = gr.File(label="Mystery (.txt)", file_types=['.txt'])
                with gr.Row():
                    candidate1 = gr.File(label="Candidate 1 (.txt)", file_types=['.txt'])
                    gt1_checkbox = gr.Checkbox(label="Ground Truth?", value=False)

                with gr.Row():
                    candidate2 = gr.File(label="Candidate 2 (.txt)", file_types=['.txt'])
                    gt2_checkbox = gr.Checkbox(label="Ground Truth?", value=False)

                with gr.Row():
                    candidate3 = gr.File(label="Candidate 3 (.txt)", file_types=['.txt'])
                    gt3_checkbox = gr.Checkbox(label="Ground Truth?", value=False)
                
                validation_msg = gr.Textbox(label="Validation Result", interactive=False)
                
            for checkbox in [gt1_checkbox, gt2_checkbox, gt3_checkbox]:
                checkbox.change(
                    fn=validate_ground_truth,
                    inputs=[gt1_checkbox, gt2_checkbox, gt3_checkbox],
                    outputs=[ground_truth_author, validation_msg]
                )

        
        # ── Load Task Button ─────────────────────────────────────
        gr.HTML(instruction_callout("Click the button below to load the tasks and generate embeddings using selected model."))
        load_button = gr.Button("Load Task & Generate Embeddings")

        # ── HTML outputs for author texts ───────────────────────────
        default_outputs = load_instance(0, instances)
        #dont need defaults since they are loaded only on click of the load button
        header  = gr.HTML()
        mystery = gr.HTML()
        mystery_state = gr.State()  # Store unformatted mystery text for later use
        with gr.Row():
            c0 = gr.HTML()
            c1 = gr.HTML()
            c2 = gr.HTML()
            c0_state = gr.State()  # Store unformatted candidate 1 text for later use
            c1_state = gr.State()  # Store unformatted candidate 2 text for later use
            c2_state = gr.State()  # Store unformatted candidate 3 text for later use
        # ── State to hold embeddings DataFrame ─────────────────────
        task_authors_embeddings_df = gr.State()  # Store embeddings of task authors
        background_authors_embeddings_df = gr.State()  # Store background authors DataFrame
        task_mode.change(
            fn=toggle_task,
            inputs=[task_mode],
            outputs=[predefined_container, custom_container]
        )
        # ── Wire call to load task and generate embeddings once load button is clicked ───────────────────
        predicted_author = gr.State()  # Store predicted author from the embeddings
        load_button.click(
            fn=lambda: gr.update(value="⏳ Loading... Please wait", interactive=False),
            inputs=[],
            outputs=[load_button]
        ).then(
            fn=lambda mode, dropdown, mystery, c1, c2, c3, ground_truth_author, model_radio, custom_model_input: 
            update_task_display(
                mode,
                dropdown,
                instances,       # closed over
                clustered_authors_df,
                mystery,
                c1,
                c2,
                c3,
                ground_truth_author,            # true_author placeholder
                model_radio,
                custom_model_input
            ),
            inputs=[task_mode, task_dropdown, mystery_input, candidate1, candidate2, candidate3, ground_truth_author, model_radio, custom_model],
            outputs=[header, mystery, c0, c1, c2, mystery_state, c0_state, c1_state, c2_state, task_authors_embeddings_df, background_authors_embeddings_df, predicted_author, ground_truth_author]  # embeddings_df is a placeholder for now
        ).then(
            fn=lambda: gr.update(value="Load Task & Generate Embeddings", interactive=True),
            inputs=[],
            outputs=[load_button]
        )

        # ── Visualization for features ─────────────────────────────
        gr.HTML(instruction_callout("Run visualization to see which author is similar to the mystery document."))
        run_btn   = gr.Button("Run visualization")
        bg_proj_state = gr.State()
        bg_lbls_state = gr.State()
        bg_authors_df = gr.State()  # Holds the background authors DataFrame
        with gr.Row():
            with gr.Column(scale=3):
                axis_ranges = gr.Textbox(visible=False, elem_id="axis-ranges")
                plot = gr.Plot(
                    label="Visualization",
                    elem_id="feature-plot",
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
                            const gd = document.querySelector('#feature-plot .js-plotly-plot');
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
                    The grey ● symbols represent the background corpus—real authors with diverse writing styles. 
                    You can zoom in on any region of the plot. The system will analyze the visible authors 
                    in that area and list the most representative writing style features for the zoomed-in region.<br>
                    Use this to compare your mystery text’s position against nearby writing styles and
                    investigate which features distinguish it from others.
                    </p>
                """
                gr.HTML(styled_html(expl_html))
        
        cluster_dropdown = gr.Dropdown(choices=[], label="Select Cluster to Inspect", visible=False)
        style_map_state = gr.State()
        llm_style_feats_analysis = gr.State()
        visible_zoomed_authors = gr.State()

        if use_cluster_feats:
            # ── Dynamic Cluster Choice dropdown ──────────────────────────────────
            gr.HTML(instruction_callout("Choose a cluster from the dropdown below to inspect whether its features appear in the mystery author’s text."))
            cluster_dropdown.visible = True
        else:
            gr.HTML(instruction_callout("Zoom in on the plot to select a set of background authors and see the presence of the top features from this set in candidate and mystery authors."))
           
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
                        LLM-derived style  features prominent in the zoomed-in region
                    </div>
                    """)
                features_rb = gr.Radio(choices=[], label="LLM-derived style features for this zoomed-in region")#, label="Features from the cluster closest to the Mystery Author", info="LLM-derived style features for this cluster")
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
                        Gram2Vec Features prominent in the zoomed-in region
                    </div>
                    """)
                gram2vec_rb    = gr.Radio(choices=[], label="Gram2Vec features for this zoomed-in region")#, label="Top-10 Gram2Vec Features most likely to occur in Mystery Author", info="Most prominent Gram2Vec features in the mystery text")
                gram2vec_state = gr.State()

        # ── Visualization button click ───────────────────────────────
        run_btn.click(
            fn=lambda iid, model_radio, custom_model_input, task_authors_embeddings_df, background_authors_embeddings_df, predicted_author, ground_truth_author: visualize_clusters_plotly(
                int(iid.replace('Task ','')), cfg, instances, model_radio,
                custom_model_input, task_authors_embeddings_df, background_authors_embeddings_df, predicted_author, ground_truth_author
            ),
            inputs=[task_dropdown, model_radio, custom_model, task_authors_embeddings_df, background_authors_embeddings_df, predicted_author, ground_truth_author],
            outputs=[plot, style_map_state, bg_proj_state, bg_lbls_state, bg_authors_df]
        )
        
        # Populate feature list based on selection. 
        if use_cluster_feats:
            # Use cluster-based flow
            cluster_dropdown.change(
                fn=on_cluster_change,
                inputs=[cluster_dropdown, style_map_state],
                outputs=[features_rb, gram2vec_rb , feature_list_state] 
                #adding feature_list_state to persisit all llm features in the app state
            )
        else:

            axis_ranges.change(
                fn=handle_zoom, 
                inputs=[axis_ranges, bg_proj_state, bg_lbls_state, bg_authors_df, task_authors_embeddings_df], 
                outputs=[features_rb, gram2vec_rb , llm_style_feats_analysis, feature_list_state, visible_zoomed_authors]
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
        show_background_checkbox = gr.Checkbox(label="Show spans in background authors", value=False)
        background_html = gr.HTML(visible=False)
        # print(f"in app: all_feats={feature_list_state.value}")
        # print(f"in app: sel_feat_llm={features_rb.value}")


        combined_btn.click(
            fn=show_combined_spans_all,
            inputs=[features_rb, 
                    gram2vec_rb, 
                    llm_style_feats_analysis, 
                    background_authors_embeddings_df, 
                    task_authors_embeddings_df, 
                    visible_zoomed_authors, 
                    predicted_author, 
                    ground_truth_author],
            outputs=[combined_html, background_html]
        )
        # mapping -->
        # iid = task_dropdown.value
        # sel_feat_llm = features_rb.value
        # all_feats = feature_list_state.value
        # sel_feat_g2v = gram2vec_rb.value
        # combined_html -> spans/html for task authors
        # background_html -> spans/html for background authors

        show_background_checkbox.change(
            fn=lambda show: gr.update(visible=show),
            inputs=[show_background_checkbox],
            outputs=[background_html]
        )

    demo.launch(share=share)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cluster_feats", action="store_true", help="Use cluster-based selection for features")
    args = parser.parse_args()
    app(share=True, use_cluster_feats=args.use_cluster_feats)
