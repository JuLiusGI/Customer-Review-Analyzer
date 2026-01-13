import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
from typing import List, Dict, Any

class ReviewAnalyzer:
    """
    Analyzer for customer reviews using Embeddings + K-Means + Sentiment Analysis.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', num_clusters: int = 3):
        """
        Initialize the analyzer with models.
        """
        print(f"Loading embedding model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        self.num_clusters = num_clusters
        
        print("Loading sentiment analysis pipeline...")
        # device=-1 for CPU, set to 0 if CUDA is available
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def process_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate embeddings and cluster reviews.
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column.")

        print("Generating embeddings...")
        embeddings = self.embedder.encode(df['text'].tolist(), show_progress_bar=True)
        
        print(f"Clustering into {self.num_clusters} topics...")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(embeddings)
        
        # Store embeddings for visualization later
        # We can't easily store numpy arrays in CSV, so we'll handle usually in memory 
        # or save separate .npy file if needed. For now, we return them or store in object.
        self.embeddings = embeddings
        
        return df

    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for each review.
        """
        print("Analyzing sentiment...")
        # Truncate to 512 tokens to match BERT limit roughly
        texts = df['text'].tolist()
        results = self.sentiment_analyzer(texts, truncation=True, max_length=512)
        
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        return df

    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate a text summary of the analysis.
        """
        report = []
        report.append("# Customer Review Analysis Report")
        report.append(f"Total Reviews: {len(df)}")
        report.append("-" * 40)
        
        # Cluster Analysis
        report.append(f"\n## Topic Clusters (K={self.num_clusters})")
        for cluster_id in range(self.num_clusters):
            cluster_df = df[df['cluster'] == cluster_id]
            size = len(cluster_df)
            percentage = (size / len(df)) * 100
            
            report.append(f"\n### Cluster {cluster_id} ({size} reviews, {percentage:.1f}%)")
            
            # Sample reviews
            samples = cluster_df['text'].sample(min(3, size), random_state=42).tolist()
            report.append("**Sample Reviews:**")
            for s in samples:
                report.append(f"- \"{s}\"")
            
            # Sentiment breakdown in cluster
            sentiment_counts = cluster_df['sentiment_label'].value_counts(normalize=True)
            report.append("\n**Sentiment Distribution:**")
            for label, prop in sentiment_counts.items():
                report.append(f"- {label}: {prop:.1f}%")

        return "\n".join(report)

if __name__ == "__main__":
    # Smoke test
    try:
        df = pd.read_csv('data/reviews.csv')
        analyzer = ReviewAnalyzer()
        df = analyzer.process_reviews(df)
        df = analyzer.analyze_sentiment(df)
        report = analyzer.generate_report(df)
        print("\n" + report)
    except Exception as e:
        print(f"Error during smoke test: {e}")
