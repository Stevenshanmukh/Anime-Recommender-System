import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from download_helper import download_model_files

# Get current directory
CURRENT_DIR = Path(__file__).parent

# Page config
st.set_page_config(
    page_title="Anime Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download model files if needed (runs once, cached)
if not download_model_files():
    st.error("Failed to download model files. Please try again or download manually.")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data with correct path
@st.cache_data
def load_data():
    df = pd.read_parquet(CURRENT_DIR / 'anime_features.parquet')
    return df

df = load_data()

# Home page
st.markdown('<div class="main-header">🎬 Anime Recommender System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Recommendations with Multi-Modal Deep Learning</div>', unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Anime", f"{len(df):,}")
with col2:
    st.metric("⚡ Search Speed", "5ms")
with col3:
    st.metric("🎯 NDCG Score", "1.0")
with col4:
    st.metric("⭐ Avg Quality", "7.93/10")

st.markdown("---")

# Features
st.header("🚀 System Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Recommendation Modes")
    st.markdown("""
    - **Single Title**: Find anime similar to one you love
    - **Multi-Title**: Combine multiple preferences
    - **Data Explorer**: Browse statistics and visualizations
    """)
    
    st.subheader("🧠 AI Technology")
    st.markdown("""
    - **Text Embeddings**: 3,573-dimensional semantic vectors
    - **Graph Features**: Network relationships (studios/producers)
    - **LightGBM Ranker**: Learning-to-rank with perfect NDCG
    - **FAISS Index**: Lightning-fast similarity search
    """)

with col2:
    st.subheader("💡 Explainability")
    st.markdown("""
    - **SHAP Values**: Feature importance analysis
    - **Natural Language**: Human-readable explanations
    - **Quality Scores**: Rating and popularity signals
    - **Content Match**: Genre and theme alignment
    """)
    
    st.subheader("📊 Quality Metrics")
    st.markdown("""
    - Average recommendation score: **7.93/10**
    - Genre relevance: **1.73 overlap**
    - Diversity: **6.2 unique genres**
    - Coverage: **345+ anime**
    """)

st.markdown("---")
st.header("⚙️ How It Works")

st.markdown("""
1. **Input**: Select an anime or describe preferences
2. **Candidate Generation**: FAISS retrieves similar anime (<5ms)
3. **Feature Extraction**: 20 features computed per candidate
4. **Ranking**: LightGBM model predicts optimal order
5. **Explanation**: SHAP values generate human-readable reasons
""")

st.markdown("---")
st.header("🎯 Get Started")

st.info("👈 Use the sidebar to navigate to different features!")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with Streamlit • Powered by LightGBM & FAISS</p>
    <p><a href="https://github.com/Stevenshanmukh/Anime-Recommender-System" target="_blank">⭐ View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
