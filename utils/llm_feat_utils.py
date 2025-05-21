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
