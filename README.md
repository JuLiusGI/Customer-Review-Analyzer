# Customer Review Analyzer

A Python tool that uses AI (Sentence Transformers & K-Means) to cluster customer reviews into topics and analyze their sentiment.

## Features

- **Text Embeddings**: Converts reviews into vector embeddings using `all-MiniLM-L6-v2`.
- **Clustering**: Groups similar reviews using K-Means clustering.
- **Sentiment Analysis**: Detects sentiment using a pre-trained DistilBERT pipeline.
- **Interactive Dashboard**: Streamlit-based Web UI for easy analysis.

## Setup

1.  **Install Dependencies**:

    ```bash
    python -m venv venv
    .\venv/Scripts/Activate
    pip install -r requirements.txt
    ```

2.  **Run Web UI (Recommended)**:

    ```bash
    streamlit run src/gui.py
    ```

3.  **Run CLI Mode (Alternative)**:
    - Generate Data: `python src/generate_data.py`
    - Run Analyzer: `python src/app.py`

## Output

Check the `output/` folder for:

- `analysis_report.txt`
- `cluster_plot.png`
- `sentiment_dist.png`
