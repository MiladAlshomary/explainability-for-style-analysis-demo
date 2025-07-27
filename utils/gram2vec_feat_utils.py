import re
import html

from collections import namedtuple
from gram2vec.feature_locator import find_feature_spans
from functools import lru_cache

from utils.llm_feat_utils import generate_feature_spans_cached

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

    return style + highlighted


def show_combined_spans_all(client, iid, selected_feature_llm, features_list, instances, selected_feature_g2v, task_mode,mystery_state, c0_state, c1_state, c2_state):
    """
    For mystery + 3 candidates:
     1. get llm spans via your existing cache+API
     2. get gram2vec spans via find_feature_spans
     3. merge and highlight both
    """
    if task_mode == "Predefined HRS Task":
        iid = int(iid)
        inst = instances[iid]

        # texts
        texts = [
        ("Mystery Author", inst['Q_fullText']),
        ("Candidate 1", inst[f'a{inst["latent_rank"][0]}_fullText']),
        ("Candidate 2",         inst[f'a{inst["latent_rank"][1]}_fullText']),
        ("Candidate 3",         inst[f'a{inst["latent_rank"][2]}_fullText']),
        ]
    else:
        # custom task
        iid = "custom" # Add some type of hashing/unique identifier here as well
        texts = [
            ("Mystery Text", mystery_state),
            ("Candidate 1", c0_state),
            ("Candidate 2", c1_state),
            ("Candidate 3", c2_state)
        ]
    

    # get llm spans map (list of spans objects) for each text
    if selected_feature_llm and selected_feature_llm != "None":
        print(f"in show spans: Selected LLM feature: {selected_feature_llm}")
        print(f"in show spans: features_list: {features_list}")
        llm_maps = [
        generate_feature_spans_cached(client, f"{iid}", texts[0][1], features_list, role="mystery"),
        generate_feature_spans_cached(client, f"{iid}_cand0",   texts[1][1], features_list, role="candidate"),
        generate_feature_spans_cached(client, f"{iid}_cand1",   texts[2][1], features_list, role="candidate"),
        generate_feature_spans_cached(client, f"{iid}_cand2",   texts[3][1], features_list, role="candidate"),
        ]
        # get span indexes for each text
        llm_spans_list = [
            [
                # positional: first arg → start_char, second → end_char
                Span(txt.find(s), txt.find(s) + len(s))
                for s in llm_maps[i].get(selected_feature_llm) 
                if s in txt
            ]
            for i, (_, txt) in enumerate(texts)
        ]
    else:
        print("Skipping LLM span extraction: feature is None")
        llm_spans_list = [[] for _ in texts]

    if selected_feature_g2v and selected_feature_g2v != "None":
        # get gram2vec spans
        gram_spans_list = []
        # key, _ = selected_feature_g2v.split(":", 1)
        # sel_g2v_short = FEATURE_HANDLERS.get(key, key)
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
    html = []
    for i, (label, txt) in enumerate(texts):
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

    return "<div>" + "\n<hr>\n".join(html) + "</div>"

