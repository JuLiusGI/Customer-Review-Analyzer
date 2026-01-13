import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.decomposition import PCA
from analyzer import ReviewAnalyzer

def main():
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # 1. Load Data
    data_path = 'data/reviews.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run src/generate_data.py first.")
        sys.exit(1)
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Run Analysis
    analyzer = ReviewAnalyzer(num_clusters=3)
    df = analyzer.process_reviews(df)
    df = analyzer.analyze_sentiment(df)

    # 3. Generate Report
    print("Generating report...")
    report = analyzer.generate_report(df)
    report_path = 'output/analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # 4. Generate Visualizations
    print("Generating visualizations...")
    
    # 4a. Cluster Plot (PCA reduced to 2D)
    # analyzer.embeddings is available because we stored it in process_reviews
    pca = PCA(n_components=2)
    coords = pca.fit_transform(analyzer.embeddings)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df['cluster'], cmap='viridis', alpha=0.7)
    plt.title('Review Clusters (PCA Reduced)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.savefig('output/cluster_plot.png')
    plt.close()
    print("Saved output/cluster_plot.png")

    # 4b. Sentiment Distribution by Cluster using a standard bar chart
    # Calculate counts
    sentiment_counts = df.groupby(['cluster', 'sentiment_label']).size().unstack(fill_value=0)
    
    # Normalize to percentages for better comparison
    sentiment_pct = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
    
    sentiment_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='RdYlGn')
    plt.title('Sentiment Distribution by Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Percentage')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig('output/sentiment_dist.png')
    plt.close()
    print("Saved output/sentiment_dist.png")

    print("\nAnalysis Complete!")

if __name__ == "__main__":
    main()
