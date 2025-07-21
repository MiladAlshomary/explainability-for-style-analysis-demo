import gradio as gr
import json

from utils.visualizations import *
from utils.llm_feat_utils import *
from utils.gram2vec_feat_utils import *
from utils.interp_space_utils import *
from utils.ui import *

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

def app(share=False, use_cluster_feats=False):
    instances, instance_ids = get_instances(cfg['instances_to_explain_path'])

    interp      = load_interp_space(cfg)
    clustered_authors_df = interp['clustered_authors_df']
    task_authors_df = instance_to_df(instances[0]) #default task is task number 0
    
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
                candidate1 = gr.File(label="Candidate1 (.txt)", file_types=['.txt'])
                candidate2 = gr.File(label="Candidate2 (.txt)", file_types=['.txt'])
                candidate3 = gr.File(label="Candidate3 (.txt)", file_types=['.txt'])

        # ── HTML outputs for author texts ───────────────────────────
        default_outputs = load_instance(0, instances)
        header  = gr.HTML(value=default_outputs[0])
        mystery = gr.HTML(value=default_outputs[1])
        mystery_state = gr.State(value=default_outputs[1])  # Store unformatted mystery text for later use
        with gr.Row():
            c0 = gr.HTML(value=default_outputs[2])
            c1 = gr.HTML(value=default_outputs[3])
            c2 = gr.HTML(value=default_outputs[4])
            c0_state = gr.State(value=default_outputs[2])  # Store unformatted candidate 1 text for later use
            c1_state = gr.State(value=default_outputs[3])  # Store unformatted candidate 2 text for later use
            c2_state = gr.State(value=default_outputs[4])  # Store unformatted candidate 3 text for later use
        # ── State to hold embeddings DataFrame ─────────────────────
        task_authors_embeddings_df = gr.State()  # Store embeddings of task authors
        background_authors_embeddings_df = gr.State()  # Store background authors DataFrame
        # ── Wire up callbacks ─────────────────────────────────────
        task_mode.change(
            fn=toggle_task,
            inputs=[task_mode],
            outputs=[predefined_container, custom_container]
        )
        # Update displayed texts for both modes
        task_mode.change(
            fn=lambda mode, dropdown, mystery, c1, c2, c3, model_radio, custom_model_input: 
            update_task_display(
                mode,
                dropdown,
                instances,       # closed over
                clustered_authors_df,
                mystery,
                c1,
                c2,
                c3,
                None,            # true_author placeholder
                model_radio,
                custom_model_input
            ),
            inputs=[task_mode, task_dropdown, mystery_input, candidate1, candidate2, candidate3, model_radio, custom_model],
            outputs=[header, mystery, c0, c1, c2, mystery_state, c0_state, c1_state, c2_state, task_authors_embeddings_df, background_authors_embeddings_df]  # embeddings_df is a placeholder for now
        )
        # When selecting a predefined task
        task_dropdown.change(
            fn=lambda iid: load_instance(int(iid.replace('Task ', '')), instances),
            inputs=[task_dropdown],
            outputs=[header, mystery, c0, c1, c2]
        )
        # When user edits custom fields
        for inp in [mystery_input, candidate1, candidate2, candidate3]:
            inp.change(
            fn=lambda mode, dropdown, mystery, c1, c2, c3, model_radio, custom_model_input: 
            update_task_display(
                mode,
                dropdown,
                instances,       # closed over
                clustered_authors_df,
                mystery,
                c1,
                c2,
                c3,
                None,            # true_author placeholder
                model_radio,
                custom_model_input
            ),
            inputs=[task_mode, task_dropdown, mystery_input, candidate1, candidate2, candidate3, model_radio, custom_model],
            outputs=[header, mystery, c0, c1, c2, mystery_state, c0_state, c1_state, c2_state, task_authors_embeddings_df, background_authors_embeddings_df]  # embeddings_df is a placeholder for now
        )

        # Whenever the radio changes, refresh the task display to compute embeddings as well
        model_radio.change(
            fn=lambda mode, dropdown, mystery, c1, c2, c3, model_radio, custom_model_input: 
            update_task_display(
                mode,
                dropdown,
                instances,       # closed over
                clustered_authors_df,
                mystery,
                c1,
                c2,
                c3,
                None,            # true_author placeholder
                model_radio,
                custom_model_input
            ),
            inputs=[task_mode, task_dropdown, mystery_input, candidate1, candidate2, candidate3, model_radio, custom_model],
            outputs=[header, mystery, c0, c1, c2, mystery_state, c0_state, c1_state, c2_state, task_authors_embeddings_df, background_authors_embeddings_df]  # embeddings_df is a placeholder for now
        )

        # And if the user types in a custom model ID, refresh the task display to compute embeddings as well
        custom_model.change(
            fn=lambda mode, dropdown, mystery, c1, c2, c3, model_radio, custom_model_input: 
            update_task_display(
                mode,
                dropdown,
                instances,       # closed over
                clustered_authors_df,
                mystery,
                c1,
                c2,
                c3,
                None,            # true_author placeholder
                model_radio,
                custom_model_input
            ),
            inputs=[task_mode, task_dropdown, mystery_input, candidate1, candidate2, candidate3, model_radio, custom_model],
            outputs=[header, mystery, c0, c1, c2,  mystery_state, c0_state, c1_state, c2_state, task_authors_embeddings_df, background_authors_embeddings_df]
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
        
        cluster_dropdown = gr.Dropdown(choices=[], label="Select Cluster to Inspect", visible=False)
        style_map_state = gr.State()

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
            fn=lambda iid, model_radio, task_mode: visualize_clusters_plotly(
                int(iid.replace('Task ','')), cfg, instances, model_radio, task_mode
            ),
            inputs=[task_dropdown, model_radio, task_mode],
            outputs=[plot, cluster_dropdown, style_map_state, bg_proj_state, bg_lbls_state, bg_authors_df]
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
                inputs=[axis_ranges, bg_proj_state, bg_lbls_state, bg_authors_df], 
                outputs=[features_rb, gram2vec_rb , feature_list_state]
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
            fn=lambda iid, sel_feat_llm, all_feats, sel_feat_g2v, task_mode, mystery_state, c0_state, c1_state, c2_state: show_combined_spans_all(
                client, iid.replace('Task ', ''), sel_feat_llm, all_feats, instances, sel_feat_g2v, task_mode, mystery_state, c0_state, c1_state, c2_state
            ),
            inputs=[task_dropdown, features_rb, feature_list_state, gram2vec_rb, task_mode, mystery_state, c0_state, c1_state, c2_state],
            outputs=[combined_html]
        )
        # mapping -->
        # iid = task_dropdown.value
        # sel_feat_llm = features_rb.value
        # all_feats = feature_list_state.value
        # sel_feat_g2v = gram2vec_rb.value

    demo.launch(share=share)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cluster_feats", action="store_true", help="Use cluster-based selection for features")
    args = parser.parse_args()
    app(share=True, use_cluster_feats=args.use_cluster_feats)
