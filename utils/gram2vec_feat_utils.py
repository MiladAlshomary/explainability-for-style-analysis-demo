import re
import html

from collections import namedtuple
from gram2vec.feature_locator import find_feature_spans
from functools import lru_cache

from utils.llm_feat_utils import generate_feature_spans_cached
import pandas as pd
Span = namedtuple('Span', ['start_char', 'end_char'])

from gram2vec import vectorizer

# ── the FEATURE_HANDLERS & loader  ────────────
FEATURE_HANDLERS = {
    "Part-of-Speech Unigram": "pos_unigrams",
    "Part-of-Speech Bigram":  "pos_bigrams",
    "Function Word":          "func_words",
    "Punctuation":            "punctuation",
    "Letter":                 "letters",
    "Dependency Label":       "dep_labels",
    "Morphology Tag":         "morph_tags",
    "Sentence Type":          "sentences",
    "Emoji":                  "emojis",
    "Number of Tokens":       "num_tokens"
}

@lru_cache(maxsize=1)
def load_code_map(txt_path: str = "utils/augmented_human_readable.txt") -> dict:
    code_map = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            human, code = [p.strip() for p in line.split(":", 1)]
            code_map[human] = code
    return code_map

def get_shorthand(feature_str: str) -> str:
    """
    Expects 'Category:Human-Readable', returns e.g. 'pos_unigrams:ADJ' or None.
    """
    try:
        category, human = [p.strip() for p in feature_str.split(":", 1)]
        # print(f"Category: {category}, Human: {human}")
    except ValueError:
        # print("Invalid format for feature string:", feature_str)
        return None
    if category not in FEATURE_HANDLERS:
        return None
    code = load_code_map().get(human)
    if code is None:
        # print(f"Warning: No code found for human-readable feature '{human}'")
        return None  # fallback to the human-readable name
    return f"{FEATURE_HANDLERS[category]}:{code}"

def get_fullform(shorthand: str) -> str:
    """
    Expects 'prefix:code' (e.g., 'pos_unigrams:ADJ'), returns 'Category:Human-Readable' 
    (e.g., 'Part-of-Speech Unigram:Adjective'), or None if invalid.
    """
    try:
        prefix, code = shorthand.split(":", 1)
    except ValueError:
        return None

    # Reverse FEATURE_HANDLERS
    reverse_handlers = {v: k for k, v in FEATURE_HANDLERS.items()}
    category = reverse_handlers.get(prefix)
    if category is None:
        return None

    # Reverse code map
    code_map = load_code_map()
    reverse_code_map = {v: k for k, v in code_map.items()}
    human = reverse_code_map.get(code)
    if human is None:
        return None

    return f"{category}:{human}"

def highlight_both_spans(text, llm_spans, gram_spans):
    """
    Walk the original `text` once, injecting <mark> tags at the correct offsets,
    so that nested or overlapping highlights never stomp on each other.
    """

    # Inline CSS : mark-llm is in yellow, mark-gram in blue
    style = """
    <style>
      .mark-llm  { background-color: #fff176; } 
      .mark-gram { background-color: #90caf9; }
    </style>
    """

    # Turn each span into two “events”: open and close
    events = []
    for s in llm_spans:
        events.append((s.start_char, 'open',  'llm'))
        events.append((s.end_char,   'close', 'llm'))
    for s in gram_spans:
        events.append((s.start_char, 'open',  'gram'))
        events.append((s.end_char,   'close', 'gram'))

    # Sort by position;
    events.sort(key=lambda e: (e[0], 0 if e[1]=='open' else 1))

    out = []
    last_idx = 0
    for idx, typ, cls in events:
        # escape the slice between last index and this event
        out.append(html.escape(text[last_idx:idx]))
        if typ == 'open':
            out.append(f'<mark class="mark-{cls}">')
        else:
            out.append('</mark>')
        last_idx = idx

    out.append(html.escape(text[last_idx:]))
    highlighted = "".join(out)

    highlighted = highlighted.replace('\n', '<br>')

    return style + highlighted


def show_combined_spans_all(selected_feature_llm, selected_feature_g2v, 
                            llm_style_feats_analysis, background_authors_embeddings_df, task_authors_embeddings_df, visible_authors, predicted_author=None, ground_truth_author=None, max_num_authors=7):
    """
    For mystery + 3 candidates:
     1. get llm spans via your existing cache+API
     2. get gram2vec spans via find_feature_spans
     3. merge and highlight both
    """
    print(f"\n\n\n\n\nShowing combined spans for LLM feature '{selected_feature_llm}' and Gram2Vec feature '{selected_feature_g2v}'")
    print(f"predicted_author: {predicted_author}, ground_truth_author: {ground_truth_author}")
    print(f" keys = {background_authors_embeddings_df.keys()}")
    
    # background_and_task_authors = pd.concat([task_authors_embeddings_df, background_authors_embeddings_df])
    # background_and_task_authors = background_and_task_authors[background_and_task_authors.authorID.isin(visible_authors)]

    #get the visible background authors
    background_authors_embeddings_df = background_authors_embeddings_df[background_authors_embeddings_df.authorID.isin(visible_authors)]
    background_and_task_authors = pd.concat([task_authors_embeddings_df, background_authors_embeddings_df])

    authors_texts = ['\n\n =========== \n\n'.join(x) if type(x) == list else x for x in background_and_task_authors[:max_num_authors]['fullText'].tolist()]
    authors_names = background_and_task_authors[:max_num_authors]['authorID'].tolist()
    print(f"Number of authors to show: {len(authors_texts)}")
    print(f"Authors names: {authors_names}")
    texts = list(zip(authors_names, authors_texts))

    if selected_feature_llm and selected_feature_llm != "None":
        # print(llm_style_feats_analysis)
        author_list = list(llm_style_feats_analysis['spans'].values())
        llm_spans_list = []
        for i, (_, txt) in enumerate(texts):
            author_spans_list = []
            for txt_span in author_list[i][selected_feature_llm]:
                    author_spans_list.append(Span(txt.find(txt_span), txt.find(txt_span) + len(txt_span)))
            llm_spans_list.append(author_spans_list)
    else:
        print("Skipping LLM span extraction: feature is None")
        llm_spans_list = [[] for _ in texts]

    if selected_feature_g2v and selected_feature_g2v != "None":
        # get gram2vec spans
        gram_spans_list = []
        print(f"Selected Gram2Vec feature: {selected_feature_g2v}")
        short = get_shorthand(selected_feature_g2v)
        print(f"short hand: {short}")
        for role, txt in texts:
            try:
                print(f"Finding spans for {short} {role}")
                spans = find_feature_spans(txt, short)
                # spans = [Span(fs.start_char, fs.end_char) for fs in raw_spans]
            except:
                print(f"Error finding spans for {short} {role}")
                spans = []
            gram_spans_list.append(spans)
    else:
        print("Skipping Gram2Vec span extraction: feature is None")
        gram_spans_list = [[] for _ in texts]

    # build HTML blocks
    print(f" ----> Number of authors: {len(texts)}")

    html_task_authors = create_html(
        texts[:4], #first 4 are task
        llm_spans_list,
        gram_spans_list,
        selected_feature_llm,
        selected_feature_g2v,
        short,
        background = False,
        predicted_author=predicted_author,
        ground_truth_author=ground_truth_author
    )
    combined_html = "<div>" + "\n<hr>\n".join(html_task_authors) + "</div>"

    html_background_authors = create_html(
        texts[4:], #last three are background
        llm_spans_list,
        gram_spans_list,
        selected_feature_llm,
        selected_feature_g2v,
        short, 
        background = True,
        predicted_author=predicted_author,
        ground_truth_author=ground_truth_author
    )
    background_html = "<div>" + "\n<hr>\n".join(html_background_authors) + "</div>"
    return combined_html, background_html

def get_label(label: str, predicted_author=None, ground_truth_author=None, bg_id: int=0) -> str:
    """
    Returns a human-readable label for the author.
    """
    print(f"get_label called with label: {label}, predicted_author: {predicted_author}, ground_truth_author: {ground_truth_author}, bg_id: {bg_id}")
    if label.startswith("Mystery") or label.startswith("Q_author"):
        return "Mystery Author"
    elif label.startswith("a0_author") or label.startswith("a1_author") or label.startswith("a2_author") or label.startswith("Candidate"):
        if label.startswith("Candidate"):
            id = int(label.split(" ")[2])  # Get the number after 'Candidate Author'
        else:
            id = label.split("_")[0][-1] # Get the last character of the first part (a0, a1, a2)
        if predicted_author is not None and ground_truth_author is not None:
            if int(id) == predicted_author and int(id) == ground_truth_author:
                return f"Candidate {int(id)+1} (Predicted & Ground Truth)"
            elif int(id) == predicted_author:
                return f"Candidate {int(id)+1} (Predicted)"
            elif int(id) == ground_truth_author:
                return f"Candidate {int(id)+1} (Ground Truth)"
            else:
                return f"Candidate {int(id)+1}"
        else:
            return f"Candidate {int(id)+1}"
    else:
        return f"Background Author {bg_id+1}"

def create_html(texts, llm_spans_list, gram_spans_list, selected_feature_llm, selected_feature_g2v, short=None, background = False, predicted_author=None, ground_truth_author=None):
    html = []
    for i, (label, txt) in enumerate(texts):
        label = get_label(label, predicted_author, ground_truth_author,  i) if background else get_label(label, predicted_author, ground_truth_author)
        combined = highlight_both_spans(txt, llm_spans_list[i], gram_spans_list[i])
        notice = ""
        if selected_feature_llm == "None":
            notice += f"""
            <div style="padding:8px; background:#eee; border:1px solid #aaa;">
              <em>No LLM feature selected.</em>
            </div>
            """
        elif not llm_spans_list[i]:
            notice += f"""
            <div style="padding:8px; background:#fee; border:1px solid #f00;">
              <em>No spans found for LLM feature "{selected_feature_llm}".</em>
            </div>
            """
        if selected_feature_g2v == "None":
            notice += f"""
            <div style="padding:8px; background:#eee; border:1px solid #aaa;">
              <em>No Gram2Vec feature selected.</em>
            </div>
            """
        elif not short:
            notice += f"""
            <div style="padding:8px; background:#fee; border:1px solid #f00;">
              <em>Invalid or unmapped feature: "{selected_feature_g2v}".</em>
            </div>
            """
        elif not gram_spans_list[i]:
            notice += f"""
            <div style="padding:8px; background:#fee; border:1px solid #f00;">
              <em>No spans found for Gram2Vec feature "{selected_feature_g2v}".</em>
            </div>
            """
        html.append(f"""
          <h3>{label}</h3>
          {notice}
          <div style="border:1px solid #ccc; padding:8px; margin-bottom:1em;">
            {combined}
          </div>
        """)
    return html