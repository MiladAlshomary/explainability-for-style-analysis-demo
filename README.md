# Authorship Attribution Explainability Tool
<div align="center">

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ExplainabiliyForAATeam/explainability-tool-for-aa)

</div>

An interactive web application for explaining and visualizing authorship attribution models. This tool provides insights into the latent space of embedding-based authorship attribution (AA) models, helping researchers and analysts understand how these models make decisions about text authorship.

## About

This demo application helps you **explore** the latent space of any sentence transformer authorship attribution model. It provides:

- **Interactive visualizations** of author embeddings and feature importance
- **Explanatory analysis** using both traditional linguistic features (Gram2Vec) and modern LLM-based features

The tool is designed for researchers, forensic linguists, and anyone interested in understanding how computational models analyze writing style and attribute authorship.

## Features

- 📊 **Interactive Gradio Interface**: User-friendly web interface for exploring authorship attribution
- 🔍 **Multi-Feature Analysis**: Compares Gram2Vec features with LLM-based stylistic analysis  
- 📈 **Visualization Tools**: t-SNE plots, embedding visualizations, and feature highlights in text
- 🤖 **OpenAI Integration**: Uses language models for advanced feature extraction and explanation
- 💾 **Caching System**: Pre-computed embeddings and features for faster analysis

## Installation

### Prerequisites

- Python 3.11+
- Git

### Clone the Repository

```bash
git clone https://github.com/MiladAlshomary/explainability-for-style-analysis-demo.git
cd explainability-for-style-analysis-demo
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies include:
- `gradio==5.30.0` - Web interface framework
- `openai` - LLM integration
- `plotly` - Interactive visualizations
- `sentence_transformers` - Text embeddings
- `gram2vec` - Linguistic feature extraction

### Environment Setup

1. Create a `.env` file in the root directory:
```bash
cp .env.example .env  
```

2. Add your OpenAI API configuration to `.env`:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # or your custom endpoint
```

## Usage

### Running the Application

To start the web application:

```bash
python app.py
```

The application will:
1. Download required datasets and cache files automatically
2. Launch a Gradio interface accessible via web browser
3. Provide an interactive dashboard for authorship analysis

### Configuration

The application behavior can be customized through `config/config.yaml`:

- **Dataset URLs**: Links to pre-computed datasets and cache files
- **Cache directories**: Local storage for downloaded data
- **Feature settings**: Control which features to use (LLM, Gram2Vec, or both)
- **Analysis parameters**: Top-k features, maximum authors, etc.

### Input Data

The tool comes with Reddit text samples by default, but provides flexibility for custom use:

- **Custom Data Upload**: Users can upload their own authorship attribution tasks in JSON format

    **Expected txt files format**:
    - Query author texts (mystery author samples)
    - Candidate author texts (known author samples)

- **Custom Models**: Compatible with any sentence transformer model


The web interface allows you to either use the pre-loaded Reddit dataset or upload your own data and specify your preferred sentence transformer model for analysis.

## Docker Deployment

A Dockerfile is included for containerized deployment:

```bash
# Build the image
docker build -t authorship-explainer .

# Run the container
docker run -p 7860:7860 authorship-explainer
```

## Project Structure

```
├── app.py                        # Main Gradio application
├── baseline_static_explanations.py  # generating static interp space
├── cluster_corpus.py             # Clustering background corpus for analysis
├── precompute_caches.py          # Precompute embeddings and feature caches
├── prepare_data.py               # Prepare samples from raw data
├── add_hf_env_to_hf_space.py     # Hugging Face Spaces environment setup
├── config/
│   └── config.yaml               # Configuration settings
├── datasets/                     # Data and cache directories
│   ├── embeddings_cache/
│   ├── feature_spans_cache/
│   └── gram2vec_cache/
├── utils/                        # Core utilities
│   ├── llm_feat_utils.py         # LLM feature extraction
│   ├── gram2vec_feat_utils.py    # Gram2Vec feature utilities
│   ├── visualizations.py         # Plotting and visualization
│   ├── interp_space_utils.py     # Embedding space analysis
│   └── ui.py                     # UI components
├── requirements.txt              # Python dependencies
└── Dockerfile                    # Container configuration
```

## Development Commands

### Prepare Training/Test Data

```bash
# Clustering the background corpus
python cluster_corpus.py ../../iarpa-hiatus/explanation_tool_files/reddit_cluster_training.pkl ../../iarpa-hiatus/explanation_tool_files/reddit_cluster_test.pkl "AnnaWegmann/Style-Embedding" ./datasets/reddit_clustered_authors.pkl --min_samples 2 --metric cosine --pca_dimensions 100 --eps 0.04

# Generate explainability sample 
python prepare_data.py ../explanation_tool_files/reddit_cluster_test.pkl ./datasets/reddit_explanation_sample.json

# Generate static explanations for a sample
python baseline_static_explanations.py generate_explanations ./datasets/reddit_explanation_sample.json ./datasets/reddit_explanation_sample_with_explanations.json --interp_space_path ./datasets/reddit_interp_space.json --model_name 'AnnaWegmann/Style-Embedding'
```

### Precompute Caches

```bash
# Precompute embeddings and feature caches
python precompute_caches.py
```

## Funding Acknowledgments

This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

