import streamlit as st
import pandas as pd
import numpy as np
import faiss
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Get parent directory (streamlit_app)
CURRENT_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Single Recommend", page_icon="🎯", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_parquet(CURRENT_DIR / 'anime_features.parquet')
    embeddings = np.load(CURRENT_DIR / 'embeddings_combined.npy')
    graph_features = pd.read_parquet(CURRENT_DIR / 'graph_features.parquet')
    return df, embeddings, graph_features

@st.cache_resource
def load_models():
    index = faiss.read_index(str(CURRENT_DIR / 'faiss_index_ivf.bin'))
    model = lgb.Booster(model_file=str(CURRENT_DIR / 'lgbm_ranker.txt'))
    return index, model

df, embeddings, graph_features = load_data()
faiss_index, lgbm_model = load_models()

st.title("🎯 Single Anime Recommendations")
st.markdown("Get personalized recommendations based on a single anime you love!")

# Simple search
search_query = st.text_input("🔍 Search by title", placeholder="e.g., Death Note, Naruto...")

if search_query:
    matches = df[df['title'].str.contains(search_query, case=False, na=False)]
    
    if len(matches) > 0:
        st.success(f"Found {len(matches)} matches!")
        
        for idx, row in matches.head(10).iterrows():
            st.write(f"**{row['title']}** - Score: {row['Score']:.2f} ({row['Type']})")
    else:
        st.warning("No anime found matching your search.")
else:
    st.info("👆 Enter an anime title to get started!")
    
    st.subheader("Popular Anime Examples")
    popular = df.nlargest(10, 'Members')[['title', 'Score', 'Type', 'Members']]
    st.dataframe(popular, use_container_width=True)
