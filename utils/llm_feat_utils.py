import json
import os
from utils.visualizations import clean_text

CACHE_DIR = "datasets/feature_spans_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def generate_feature_spans(client, text: str, features: list[str]) -> str:
    print("Calling OpenAI to extract spans")
    """
    Call to OpenAI to extract spans. Returns a JSON string.
    """
    prompt = f"""You are a linguistic specialist. Given a writing sample and a list of descriptive features, identify the exact text spans that demonstrate each feature.
    
    Important:
    - The headers like "Document 1:" etc are NOT part of the original text — ignore them.
    - For each feature, even if there is no match, return an empty list.
    - Only return exact phrases from the text.

    Respond in JSON format like:
    {{
      "feature1": ["span1", "span2"],
      "feature2": [],
      …
    }}

    Text:
    \"\"\"{text}\"\"\"

    Style Features:
    {features}
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

def generate_feature_spans_cached(client, instance_id: str, text: str, features: list[str], role: str = "mystery" ) -> dict:
    """
    Computes a cache key from instance_id + text + feature list,
    then either loads or calls the API and saves to disk.
    Returns the parsed JSON dict mapping feature->list[spans].
    """
    print(f"Generating spans for {instance_id} ({role})")
    cache_path = os.path.join(CACHE_DIR, f"{instance_id}_{role}.json")
    if os.path.exists(cache_path):
        return json.load(open(cache_path))
    else:
        raw = generate_feature_spans(client, text, features)
        mapping = json.loads(raw)
        with open(cache_path, "w") as f:
            json.dump(mapping, f, indent=2)
        return mapping

# def highlight_spans(text: str, selected_feature: str, spans_map: dict) -> str:
#     spans = spans_map.get(selected_feature, [])
#     if not spans:
#         return None  # or an empty string flagging “not present”
#     for span in spans:
#         text = text.replace(span, f"<mark>{clean_text(span)}</mark>")
#     return text

# def show_both_spans(client, iid, selected_feature, features_list, instances, cfg):
#     iid = int(iid)
#     inst = instances[iid]
#     mystery_text = inst['Q_fullText']
#     # candidate text of the predicted author
#     pred_idx     = inst['latent_rank'][0]
#     candidate_text = inst[f'a{pred_idx}_fullText']

#     # candidate text of the other two authors
#     idx_1     = inst['latent_rank'][1]
#     candidate_idx_1_text = inst[f'a{idx_1}_fullText']
#     idx_2     = inst['latent_rank'][2]
#     candidate_idx_2_text = inst[f'a{idx_2}_fullText']

#     # generate (or load) spans mapping for both texts
#     all_feats = features_list#(iid, cfg, instances)
#     #TODO: add abutton to dynaically get features from different clusters not just predicted clusters
#     mystery_map   = generate_feature_spans_cached(client, str(iid), mystery_text, all_feats, role="mystery")
#     candidate_map = generate_feature_spans_cached(client, f"{iid}_cand{pred_idx}", candidate_text, all_feats, role="candidate")
#     candidate_map_1 = generate_feature_spans_cached(client, f"{iid}_cand{idx_1}", candidate_idx_1_text, all_feats, role="candidate")
#     candidate_map_2 = generate_feature_spans_cached(client, f"{iid}_cand{idx_2}", candidate_idx_2_text, all_feats, role="candidate")

#     # highlight
#     myst = highlight_spans(mystery_text, selected_feature, mystery_map)
#     cand = highlight_spans(candidate_text, selected_feature, candidate_map)
#     cand_1 = highlight_spans(candidate_idx_1_text, selected_feature, candidate_map_1)
#     cand_2 = highlight_spans(candidate_idx_2_text, selected_feature, candidate_map_2)

#     # build HTML, handling “not present” cases
#     html_parts = []
#     html_parts.append("<h3>Mystery Author</h3>")
#     if myst is None:
#         # html_parts.append(f"<p><em>Feature “{selected_feature}” not found.</em></p>")
#         html_parts.append(f"""
#         <div style="padding:10px; margin-top:10px;">
#           <strong style="color:red;">
#             Feature “{selected_feature}” not present in text.
#           </strong>
#         </div>
#         """)
#         html_parts.append(f"<p>{clean_text(mystery_text)}</p>")
#     else:
#         html_parts.append(f"<p>{myst}</p>")

#     html_parts.append(f"<hr><h3>Predicted Candidate - C{pred_idx+1} </h3>")
#     if cand is None:
#         # html_parts.append(f"<p><em>Feature “{selected_feature}” not found.</em></p>")
#         html_parts.append(f"""
#         <div style="padding:10px; margin-top:10px;">
#           <strong style="color:red;">
#             Feature “{selected_feature}” not present in text.
#           </strong>
#         </div>
#         """)
#         html_parts.append(f"<p>{clean_text(candidate_text)}</p>")
#     else:
#         html_parts.append(f"<p>{cand}</p>")

#     html_parts.append(f"<hr><h3> Candidate - C{idx_1+1} </h3>")
#     if cand_1 is None:
#         # html_parts.append(f"<p><em>Feature “{selected_feature}” not found.</em></p>")
#         html_parts.append(f"""
#         <div style="padding:10px; margin-top:10px;">
#           <strong style="color:red;">
#             Feature “{selected_feature}” not present in text.
#           </strong>
#         </div>
#         """)
#         html_parts.append(f"<p>{clean_text(candidate_idx_1_text)}</p>")
#     else:
#         html_parts.append(f"<p>{cand_1}</p>")

#     html_parts.append(f"<hr><h3> Candidate - C{idx_2+1} </h3>")
#     if cand_2 is None:
#         # html_parts.append(f"<p><em>Feature “{selected_feature}” not found.</em></p>")
#         html_parts.append(f"""
#         <div style="padding:10px; margin-top:10px;">
#           <strong style="color:red;">
#             Feature “{selected_feature}” not present in text.
#           </strong>
#         </div>
#         """)
#         html_parts.append(f"<p>{clean_text(candidate_idx_2_text)}</p>")
#     else:
#         html_parts.append(f"<p>{cand_2}</p>")
#     return "<div style='padding:10px;border:1px solid #ccc;'>" + "\n".join(html_parts) + "</div>"


# def show_spans(client, iid, selected_feature, features_list, instances, cfg):
#     """
#     1) Compute the full feature list
#     2) Load or call the span generator
#     3) Highlight the spans in the mystery text
#     """
#     iid = int(iid)
#     inst = instances[iid]
#     text = inst['Q_fullText']
#     # all features used to generate the spans:
#     all_feats = features_list #closest_cluster_features(instance_id, cfg, instances)
#     spans_map = generate_feature_spans_cached(client, str(iid), text, all_feats)
#     spans = spans_map.get(selected_feature, [])

#     # Build the base text (with highlights if any)
#     displayed_text = text
#     if spans:
#         for span in spans:
#             displayed_text = displayed_text.replace(span, f"<mark>{span}</mark>")

#     # Wrap the text in a container
#     text_html = f"""
#     <div style="border:1px solid #ccc; padding:10px; margin-top:10px;">
#       <h3>Mystery Author Text</h3>
#       <p>{displayed_text}</p>
#     </div>
#     """

#     # If no spans, show a warning message above
#     if not spans:
#         msg_html = f"""
#         <div style="border:1px solid #f00; padding:10px; margin-top:10px; background:#fee;">
#           <strong>Feature “{selected_feature}” not present in text.</strong>
#         </div>
#         """
#         return msg_html + text_html
    
#     return text_html

   