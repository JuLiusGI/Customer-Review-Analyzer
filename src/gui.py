import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path so we can import from src if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import ReviewAnalyzer
from sklearn.decomposition import PCA

# Page Config
st.set_page_config(page_title="Customer Review Analyzer", layout="wide")

def main():
    st.title("📊 Customer Review Analyzer")
    st.markdown("Upload your review CSV to automatically analyze topics and sentiment.")

    # Sidebar for File Upload
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        st.info("CSV must have a 'text' column.")
        
        # Option to exclude sentiment (speed up)
        # analyze_sentiment = st.checkbox("Run Sentiment Analysis?", value=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column!")
                return
            
            st.success(f"Loaded {len(df)} reviews.")
            
            if st.button("Run Analysis"):
                with st.spinner("Analyzing reviews... this may take a moment"):
                    # Initialize Analyzer
                    # Cache the model loading to speed up re-runs?
                    # For simplicity, we just run it. Streamlit caching can be added later.
                    analyzer = ReviewAnalyzer(num_clusters=3)
                    
                    # 1. Process (Embeddings + Clustering)
                    df = analyzer.process_reviews(df)
                    
                    # 2. Sentiment
                    df = analyzer.analyze_sentiment(df)
                    
                    st.session_state['results'] = df
                    st.session_state['analyzer'] = analyzer
                    st.success("Analysis Complete!")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Display Results if available
    if 'results' in st.session_state:
        df = st.session_state['results']
        analyzer = st.session_state['analyzer']
        
        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", len(df))
        
        pos_count = len(df[df['sentiment_label'] == 'POSITIVE'])
        neg_count = len(df[df['sentiment_label'] == 'NEGATIVE'])
        avg_sentiment = (pos_count / len(df)) * 100
        col2.metric("Positive Sentiment", f"{avg_sentiment:.1f}%")
        
        top_cluster = df['cluster'].mode()[0]
        col3.metric("Largest Cluster Group", f"Cluster {top_cluster}")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Clusters", "Sentiment", "Raw Data"])
        
        with tab1:
            st.subheader("Analysis Report")
            report = analyzer.generate_report(df)
            st.text(report)
            
        with tab2:
            st.subheader("Topic Clusters (PCA Visualization)")
            if hasattr(analyzer, 'embeddings'):
                pca = PCA(n_components=2)
                coords = pca.fit_transform(analyzer.embeddings)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=df['cluster'], cmap='viridis', alpha=0.7)
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                plt.colorbar(scatter, label='Cluster ID')
                st.pyplot(fig)
            else:
                st.warning("Embeddings not found.")
                
        with tab3:
            st.subheader("Sentiment Distribution by Cluster")
            sentiment_counts = df.groupby(['cluster', 'sentiment_label']).size().unstack(fill_value=0)
            sentiment_pct = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sentiment_pct.plot(kind='bar', stacked=True, ax=ax, colormap='RdYlGn')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Percentage')
            st.pyplot(fig)

        with tab4:
            st.subheader("Data Table")
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results CSV",
                csv,
                "review_analysis_results.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == "__main__":
    main()
