# Author Attribution Explainability Tool

## Prerequisites
* Python 3.8 or higher
* An OpenAI API key with access to GPT-4

## Installation
1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **OpenAI API Key**: Set your key in the .env file with the following name:

   ```bash
   OPENAI_API_KEY=xyz
   ```

3. **Cache directory**: 
    - A folder `datasets/feature_spans_cache/` will be created automatically created to store OpenAI span outputs.
    - A file `datasets/tsne_cache.pkl` will be automatically created to store TSNE feature transformations.


## Running the App

Launch the Gradio app:

```bash
python app.py
```

By default, the app will open in your browser at `http://localhost:7860`.
(To run it on an HPC cluster, you would need to expose this port publicly or set up appropriate prot forwarding.)
(or use the live public link to visualize the changes)
To share publicly, set `share=True` in the `app(share=True)` call or modify in `app.py`.

## Currently Supported Features

1. **Select Instance**: Choose an ID from the dropdown to load mystery and candidate texts.
2. **Run Visualization**: Click to compute t‑SNE and display an interactive Plotly scatter plot of embeddings (Might be slow if the transofrmation is being computed for the 1st time).
3. **Choose Feature**: Select a top‑k style feature for the closest cluster from the radio buttons below the plot.
4. **Show Feature Spans**: Click to generate or load cached span annotations and highlight matching phrases in the mystery text (This might take a few seconds for the initial load).

## Caching

All OpenAI API calls for span generation are cached per `instance_id` in `./feature_spans_cache/{instance_id}_{role}*.json`.
This avoids repeated API usage for the same inputs.

## Project Structure

```
├── app.py
├── utils/           
│   ├── visualisations.py <-- helper functions for the visualizations.
├── config/            
│   ├── config.yaml  <-- holds the paths to dataset folders, gram2vec vs llm_feat flags etc. 
├── requirements.txt
├── datasets/
│   ├── hrs_explanations.json
│   └── interp/
│   │    ├── interpretable_space.pkl
│   │    ├── interpretable_space_representations.json
│   │    ├── gram2vec_feats.csv
│   │    └── train_authors.pkl
│   ├── feature_spans_cache/  <-- auto-generated cache files for openai 
│   ├── tsne_cache.pkl       <-- auto-generated cache files for tsne
```
