# Customer Review Analyzer

A Python tool that uses AI (Sentence Transformers & K-Means) to cluster customer reviews into topics and analyze their sentiment.

## Features

- **Text Embeddings**: Converts reviews into vector embeddings using `all-MiniLM-L6-v2`.
- **Clustering**: Groups similar reviews using K-Means clustering.
- **Sentiment Analysis**: Detects sentiment using a pre-trained DistilBERT pipeline.
- **Reporting**: Generates a text summary and visualization plots.

## Setup

1.  **Install Dependencies**:

    ```bash
    python -m venv venv
    .\venv\Scripts\Activate
    pip install -r requirements.txt
    ```

2.  **Generate Data** (Optional):

    ```bash
    python src/generate_data.py
    ```

3.  **Run Analyzer**:
    ```bash
    python src/app.py
    ```

## Output

Check the `output/` folder for:

- `analysis_report.txt`
- `cluster_plot.png`
- `sentiment_dist.png`
