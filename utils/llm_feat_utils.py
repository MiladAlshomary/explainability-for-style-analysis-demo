import json
import os
import hashlib
import time
from json import JSONDecodeError

CACHE_DIR = "datasets/feature_spans_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
import pandas as pd

#read and create the Gram2Vec feature set once
_g2v_df      = pd.read_csv("datasets/gram2vec_feats.csv")
GRAM2VEC_SET = set(_g2v_df['gram2vec_feats'].unique())
MAX_ATTEMPTS = 3
WAIT_SECONDS = 2

# Bump this whenever there is a change prompt, feature space, etc...
CACHE_VERSION = 2

def _feat_hash(feature: str, text: str) -> str:
    blob = json.dumps({
        "version": CACHE_VERSION,
        "text": text,
        "features": sorted(feature)
    }, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()


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

def generate_feature_spans_with_retries(client, text: str, features: list[str]) -> dict:
    """
    Calls `generate_feature_spans` with retries on failure.
    Returns the parsed JSON dict mapping feature->list[spans].
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            response_str = generate_feature_spans(client, text, features)
            result = json.loads(response_str)
            return result
        except (JSONDecodeError, ValueError) as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < MAX_ATTEMPTS - 1:
                wait_sec = WAIT_SECONDS * (2 ** attempt)
                print(f"Retrying after {wait_sec} seconds...")
                time.sleep(wait_sec)
    raise RuntimeError("All retry attempts failed for OpenAI call.")


def generate_feature_spans_cached(client, text: str, features: list[str], role: str = "mystery" ) -> dict:
    """
    Computes a cache key from text + feature list,
    then either loads or calls the API and saves to disk.
    Returns the parsed JSON dict mapping feature->list[spans].
    """
    print(f"Generating spans for ({role})")
    # print(f"feature list {features}")
    role = role.replace(" ", "_").replace("/", "_").replace("-", "_")
    print(f"Cache dir: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{role}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache: dict[str, dict] = json.load(f)
    else:
        cache = {}
    result: dict[str, list[str]] = {}
    missing_feats: list[str] = []

    for feat in features:
        if feat == "None":
            result[feat] = []
            continue
        
        h = _feat_hash(feat, text)
        if h in cache:
            result[feat] = cache[h]["spans"]
        else:
            missing_feats.append(feat)

    if missing_feats:

        mapping = generate_feature_spans_with_retries(client, text, missing_feats)
        # 4) update cache & result for each missing feature
        for feat in missing_feats:
            h = _feat_hash(feat, text)
            spans = mapping.get(feat) 
            cache[h] = {
                "feature": feat,
                "spans": spans
            }
            result[feat] = spans

        # 5) write back the combined cache
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    return result


def split_features(all_feats):
    """
    Given a list of mixed features, returns two lists:
    - llm_feats: those NOT in the Gram2Vec CSV
    - g2v_feats: those present in the CSV
    """
    g2v_feats = [feat for feat in all_feats if feat in GRAM2VEC_SET]
    llm_feats = [feat for feat in all_feats if feat not in GRAM2VEC_SET]
    return llm_feats, g2v_feats