from gram2vec.feature_locator import find_feature_spans
from gram2vec import vectorizer
import re

def highlight_gram2vec_spans(text, feature_name):
    try:
        spans = find_feature_spans(text, feature_name)
    except Exception:
        spans = []

    if not spans:
        return None

    # Sort spans in reverse order to avoid index shifting
    spans = sorted(spans, key=lambda s: s.start_char, reverse=True)
    for span in spans:
        text = (
            text[:span.start_char]
            + f"<mark>{text[span.start_char:span.end_char]}</mark>"
            + text[span.end_char:]
        )
    return text

def show_gram2vec_spans_all(iid, selected_feature, instances):
    iid = int(iid)
    inst = instances[iid]

    mystery_text = inst['Q_fullText']
    pred_idx     = inst['latent_rank'][0]
    cand1_text   = inst[f'a{pred_idx}_fullText']
    idx_2        = inst['latent_rank'][1]
    idx_3        = inst['latent_rank'][2]
    cand2_text   = inst[f'a{idx_2}_fullText']
    cand3_text   = inst[f'a{idx_3}_fullText']

    mystery_out = highlight_gram2vec_spans(mystery_text, selected_feature)
    cand1_out   = highlight_gram2vec_spans(cand1_text, selected_feature)
    cand2_out   = highlight_gram2vec_spans(cand2_text, selected_feature)
    cand3_out   = highlight_gram2vec_spans(cand3_text, selected_feature)

    html = []

    def author_block(name, original_text, highlighted_text):
        if highlighted_text is None:
            return f"""
            <h3>{name}</h3>
            <div style="padding:10px; border:1px solid #ccc;">
              <strong style="color:red;">Feature “{selected_feature}” not present.</strong>
              <p>{original_text}</p>
            </div>
            """
        return f"""
        <h3>{name}</h3>
        <div style="padding:10px; border:1px solid #ccc;">
          <p>{highlighted_text}</p>
        </div>
        """

    html.append(author_block("Mystery Author", mystery_text, mystery_out))
    html.append(author_block(f"Predicted Candidate - C{pred_idx+1}", cand1_text, cand1_out))
    html.append(author_block(f"Candidate - C{idx_2+1}", cand2_text, cand2_out))
    html.append(author_block(f"Candidate - C{idx_3+1}", cand3_text, cand3_out))

    return "<div style='margin-top:10px'>" + "\n<hr>\n".join(html) + "</div>"
