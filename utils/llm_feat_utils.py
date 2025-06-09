import json
import os
import re

CACHE_DIR = "datasets/feature_spans_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
import pandas as pd

#read and create the Gram2Vec feature set once
_g2v_df      = pd.read_csv("datasets/gram2vec_feats.csv")
GRAM2VEC_SET = set(_g2v_df['gram2vec_feats'].unique())


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


def split_features(all_feats):
    """
    Given a list of mixed features, returns two lists:
    - llm_feats: those NOT in the Gram2Vec CSV
    - g2v_feats: those present in the CSV
    """
    g2v_feats = [feat for feat in all_feats if feat in GRAM2VEC_SET]
    llm_feats = [feat for feat in all_feats if feat not in GRAM2VEC_SET]
    return llm_feats, g2v_feats

