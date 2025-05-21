import re
import html

from collections import namedtuple
from gram2vec.feature_locator import find_feature_spans
from gram2vec import vectorizer

from utils.llm_feat_utils import generate_feature_spans_cached

Span = namedtuple('Span', ['start_char', 'end_char'])

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

    print(events)

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


def show_combined_spans_all(client, iid, selected_feature_llm, features_list, instances, selected_feature_g2v):
    """
    For mystery + 3 candidates:
     1. get llm spans via your existing cache+API
     2. get gram2vec spans via find_feature_spans
     3. merge and highlight both
    """
    iid = int(iid)
    inst = instances[iid]

    # texts
    texts = [
      ("Mystery Author", inst['Q_fullText']),
      ("Predicted Candidate", inst[f'a{inst["latent_rank"][0]}_fullText']),
      ("Candidate 2",         inst[f'a{inst["latent_rank"][1]}_fullText']),
      ("Candidate 3",         inst[f'a{inst["latent_rank"][2]}_fullText']),
    ]

    # get llm spans map (list of spans objects) for each text
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
            for s in llm_maps[i].get(selected_feature_llm, [])
            if s in txt
        ]
        for i, (_, txt) in enumerate(texts)
    ]

    # get gram2vec spans
    gram_spans_list = []
    for role, txt in texts:
        try:
            print(f"Finding spans for {selected_feature_g2v} {role}")
            spans = find_feature_spans(txt, selected_feature_g2v)
        except:
            spans = []
        gram_spans_list.append(spans)

    # build HTML blocks
    html = []
    for i, (label, txt) in enumerate(texts):
        combined = highlight_both_spans(txt, llm_spans_list[i], gram_spans_list[i])
        notice = ""
        if not llm_spans_list[i]:
            notice += f"""
            <div style="padding:8px; background:#fee; border:1px solid #f00;">
              <em>No "{selected_feature_llm}" spans found.</em>
            </div>
            """
        if not gram_spans_list[i]:
            notice += f"""
            <div style="padding:8px; background:#fee; border:1px solid #f00;">
              <em>No "{selected_feature_g2v}" spans found.</em>
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
