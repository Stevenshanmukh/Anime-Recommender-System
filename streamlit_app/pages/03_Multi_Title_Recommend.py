import streamlit as st
import pandas as pd
import numpy as np
import faiss
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Multi-Title Recommend", page_icon="🎭", layout="wide")

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

st.title("🎭 Multi-Title Recommendations")
st.markdown("Combine multiple anime to get recommendations that match your diverse tastes!")

# Initialize session state
if 'selected_anime' not in st.session_state:
    st.session_state.selected_anime = []

st.subheader("Build Your Profile")

col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input("🔍 Search anime to add", placeholder="Search by title...")

with col2:
    st.write("")
    st.write("")
    if st.button("Clear All", type="secondary"):
        st.session_state.selected_anime = []
        st.rerun()

if search_query:
    matches = df[df['title'].str.contains(search_query, case=False, na=False)]
    
    if len(matches) > 0:
        st.write("**Search Results:**")
        
        for idx, row in matches.head(10).iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{row['title']}** ({row['Type']}, {row['Score']:.2f})")
            
            with col2:
                st.write(f"{', '.join(row['genres_list'][:2])}")
            
            with col3:
                if idx not in st.session_state.selected_anime:
                    if st.button("➕ Add", key=f"add_{idx}"):
                        if len(st.session_state.selected_anime) < 5:
                            st.session_state.selected_anime.append(idx)
                            st.rerun()
                        else:
                            st.warning("Maximum 5 anime allowed")

st.markdown("---")
st.subheader("Your Selected Anime")

if st.session_state.selected_anime:
    for idx in st.session_state.selected_anime:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"**{df.loc[idx, 'title']}**")
        
        with col2:
            st.write(f"Score: {df.loc[idx, 'Score']:.2f}")
        
        with col3:
            if st.button("❌ Remove", key=f"remove_{idx}"):
                st.session_state.selected_anime.remove(idx)
                st.rerun()
    
    if len(st.session_state.selected_anime) >= 2:
        st.info(f"✅ {len(st.session_state.selected_anime)} anime selected. Ready to generate recommendations!")
    else:
        st.info("👆 Select at least 2 anime to generate recommendations")
else:
    st.info("👆 Search and add anime to your profile to get started!")
    
    st.subheader("💡 Suggestion: Try These Combinations")
    st.markdown("""
    **Action Lovers:** Attack on Titan + Fullmetal Alchemist: Brotherhood
    
    **Mystery Fans:** Death Note + Steins;Gate
    
    **Slice of Life:** Your Lie in April + Clannad
    """)
