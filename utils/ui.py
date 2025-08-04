import gradio as gr
import pandas as pd
from utils.visualizations import load_instance, get_instances, clean_text
from utils.interp_space_utils import cached_generate_style_embedding, instance_to_df, compute_g2v_features


# ── Global CSS to be prepended to every block ─────────────────────────────────
GLOBAL_CSS = """
<style>
  /* Bold only the top‐level field labels (not every label) */
  .gradio-container .input_label {
    font-weight: 600 !important;
    font-size: 1.1em !important;
  }

  /* Reset radio‐option labels to normal weight/size */
  .gradio-container .radio-container .radio-option-label {
    font-weight: normal !important;
    font-size: 1em !important;
  }
  /* Give HTML output blocks a stronger border and padding */
  .gradio-container .output-html {
    border: 2px solid #888 !important;
    border-radius: 4px !important;
    padding: 0.5em !important;
    margin-bottom: 1em !important;
    font-size: 1em !important;
    line-height: 1.4 !important;
  }
</style>
"""

def styled_block(content: str) -> str:
    """
    Injects GLOBAL_CSS before the provided content.
    Returns a single HTML blob safe to pass into gr.HTML().
    """
    return GLOBAL_CSS + "\n" + content

def styled_html(html_content: str) -> str:
    """
    Wraps raw HTML content with global CSS. Pass the result to gr.HTML().
    """
    return styled_block(html_content)

def instruction_callout(text: str) -> str:
    """
    Returns a full HTML string (with global CSS) rendering `text`
    as a bold, full-width callout box.
    
    Usage:
        gr.HTML(instruction_callout(
            "Run visualization to see which author cluster contains the mystery document."
        ))
    """
    callout = f"""
    <div style="
      background: #e3f2fd;                /* light blue background */
      border-left: 5px solid #2196f3;     /* bold accent stripe */
      padding: 12px 16px;
      margin-bottom: 12px;
      font-weight: 600;
      font-size: 1.1em;
    ">
      {text}
    </div>
    """
    return styled_html(callout)

def read_txt(f):
    if not f:
        return ""
    path = f.name if hasattr(f, 'name') else f
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return fh.read().strip()
    except Exception:
        return "(Could not read file)"

# Toggle which input UI is visible
def toggle_task(mode):
    print(mode)
    return (
        gr.update(visible=(mode == "Predefined HRS Task")),
        gr.update(visible=(mode == "Upload Your Own Task"))
    )

# Update displayed texts based on mode
def update_task_display(mode, iid, instances, background_df, mystery_file, cand1_file, cand2_file, cand3_file, true_author, model_radio, custom_model_input):
    model_name = model_radio if model_radio != "Other" else custom_model_input
    if mode == "Predefined HRS Task":
        iid = int(iid.replace('Task ', ''))
        data = instances[iid]
        predicted_author = data['latent_rank'][0]
        ground_truth_author = data['gt_idx']
        mystery_txt = data['Q_fullText']
        c1_txt = data['a0_fullText']
        c2_txt = data['a1_fullText']
        c3_txt = data['a2_fullText']
        candidate_texts = [c1_txt, c2_txt, c3_txt]

        header_html, mystery_html, candidate_htmls = task_HTML(mystery_txt, candidate_texts, predicted_author, ground_truth_author)
        #create a dataframe of the task authors
        task_authors_df  = instance_to_df(instances[iid])
    else:
        header_html = "<h3>Custom Uploaded Task</h3>"
        mystery_txt = read_txt(mystery_file)
        c1_txt = read_txt(cand1_file)
        c2_txt = read_txt(cand2_file)
        c3_txt = read_txt(cand3_file)
        candidate_texts = [c1_txt, c2_txt, c3_txt]
        predicted_author = None  # Placeholder for predicted author
        header_html, mystery_html, candidate_htmls = task_HTML(mystery_txt, candidate_texts, predicted_author, true_author)
        task_authors_df  = instance_to_df(instances[iid])
    #try:
    # Generate the embeddings for the custom task authors
    # task_authors_df = generate_style_embedding(task_authors_df, 'fullText', model_name)
    # # Generate the new embedding of all the background_df authors
    # background_df = generate_style_embedding(background_df, 'fullText', model_name)
    # print(f"Generated embeddings for {len(background_df)} texts using model '{model_name}'")
    print(f"Generating embeddings for {model_name} on task authors")
    task_authors_df = cached_generate_style_embedding(task_authors_df, 'fullText', model_name)
    # Generate the new embedding of all the background_df authors
    print(f"Generating embeddings for {model_name} on background corpus")
    background_df = cached_generate_style_embedding(background_df, 'fullText', model_name)
    print(f"Generated embeddings for {len(background_df)} texts using model '{model_name}'")

    # computing g2v features
    print("Generating g2v features for on background corpus")
    background_g2v, task_authors_g2v = compute_g2v_features(background_df, task_authors_df)
    background_df['g2v_vector'] = background_g2v
    task_authors_df['g2v_vector'] = task_authors_g2v
    print(f"Gram2Vec feature generation complete")

    print(background_df.columns)
    # except Exception as e:
    #     print(f"Embedding generation failed: {e}")
    
    return [
        header_html,
        mystery_html,
        candidate_htmls[0],
        candidate_htmls[1],
        candidate_htmls[2],
        mystery_txt,
        c1_txt,
        c2_txt,
        c3_txt,
        task_authors_df,
        background_df,
    ]

def task_HTML(mystery_text, candidate_texts, predicted_author, ground_truth_author):
    header_html = f"""
    <div style="border:1px solid #ccc; padding:10px; margin-bottom:10px;">
      <h3>Here’s the mystery passage alongside three candidate texts—look for the green highlight to see the predicted author.</h3>
    </div>
    """
    mystery_text = clean_text(mystery_text)
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
        text = candidate_texts[i]
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
    return header_html, mystery_html, candidate_htmls

def toggle_custom_model(choice):
    return gr.update(visible=(choice == "Other"))