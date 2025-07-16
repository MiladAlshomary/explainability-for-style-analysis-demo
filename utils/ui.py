import gradio as gr
from utils.visualizations import load_instance


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

# Toggle which input UI is visible
def toggle_task(mode):
    print(mode)
    return (
        gr.update(visible=(mode == "Predefined HRS Task")),
        gr.update(visible=(mode == "Upload Your Own Task"))
    )

# Update displayed texts based on mode
def update_task_display(mode, iid, mystery_txt, cand1, cand2, cand3):
    if mode == "Predefined HRS Task":
        return load_instance(int(iid.replace('Task ', '')), instances)
    else:
        header_html = "<h3>Custom Uploaded Task</h3>"
        def wrap(text):
            return f"<div style='padding:0.5em; border:1px solid #ddd; margin:0.2em 0;'>{text}</div>"
        mystery_html = wrap(mystery_txt)
        cand_htmls = [wrap(t) for t in (cand1, cand2, cand3)]
        return [header_html, mystery_html] + cand_htmls