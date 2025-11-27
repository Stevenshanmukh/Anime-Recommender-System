import streamlit as st
import pandas as pd
import numpy as np
import faiss
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Multi-Title Recommend", page_icon="🎭", layout="wide")

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

# Feature extraction
def create_ranking_features(query_idx, candidate_idx):
    query_emb = embeddings[query_idx].reshape(1, -1)
    cand_emb = embeddings[candidate_idx].reshape(1, -1)
    cosine_sim = cosine_similarity(query_emb, cand_emb)[0][0]
    
    cand_graph = graph_features.loc[candidate_idx].values
    cand_score = df.loc[candidate_idx, 'Score'] if pd.notna(df.loc[candidate_idx, 'Score']) else 6.5
    cand_members = df.loc[candidate_idx, 'log_members']
    cand_favorites = df.loc[candidate_idx, 'log_favorites']
    cand_popularity = df.loc[candidate_idx, 'members_percentile']
    
    query_genres = set(df.loc[query_idx, 'genres_list'])
    cand_genres = set(df.loc[candidate_idx, 'genres_list'])
    genre_overlap = len(query_genres & cand_genres)
    genre_jaccard = len(query_genres & cand_genres) / len(query_genres | cand_genres) if len(query_genres | cand_genres) > 0 else 0
    
    query_studios = set(df.loc[query_idx, 'studios_list'])
    cand_studios = set(df.loc[candidate_idx, 'studios_list'])
    studio_overlap = len(query_studios & cand_studios)
    
    has_score = 1 if pd.notna(df.loc[candidate_idx, 'Score']) else 0
    is_highly_rated = df.loc[candidate_idx, 'is_highly_rated']
    
    return np.array([
        cosine_sim, *cand_graph, cand_score, cand_members, cand_favorites,
        cand_popularity, genre_overlap, genre_jaccard, studio_overlap,
        has_score, is_highly_rated
    ])

def get_multi_recommendations(query_indices, k=10):
    """Get recommendations based on multiple anime"""
    # Average embeddings of input anime
    query_embs = embeddings[query_indices]
    avg_query_emb = np.mean(query_embs, axis=0).reshape(1, -1)
    
    # Get candidates from FAISS
    n_candidates = k * 5
    distances, indices = faiss_index.search(avg_query_emb, n_candidates + len(query_indices))
    
    # Remove input anime
    mask = np.isin(indices[0], query_indices, invert=True)
    candidates = indices[0][mask][:n_candidates]
    
    # Use first query anime for feature extraction (approximation)
    primary_query = query_indices[0]
    
    candidate_features = np.array([
        create_ranking_features(primary_query, cand_idx)
        for cand_idx in candidates
    ])
    
    scores = lgbm_model.predict(candidate_features)
    top_k_idx = np.argsort(scores)[::-1][:k]
    
    return candidates[top_k_idx], scores[top_k_idx]

# UI
st.title("🎭 Multi-Title Recommendations")

st.markdown("""
Combine multiple anime to get recommendations that match your diverse tastes!
Select 2-5 anime and the system will find anime that share characteristics with your selection.
""")

# Initialize session state
if 'selected_anime' not in st.session_state:
    st.session_state.selected_anime = []

# Search and add anime
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

# Search results
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

# Display selected anime
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
    
    # Get recommendations
    if len(st.session_state.selected_anime) >= 2:
        st.markdown("---")
        st.subheader("🎬 Recommended Anime")
        
        num_recs = st.slider("Number of recommendations", 5, 20, 10)
        
        if st.button("Generate Recommendations", type="primary"):
            with st.spinner("Generating personalized recommendations..."):
                recs, scores = get_multi_recommendations(st.session_state.selected_anime, k=num_recs)
            
            st.success(f"Found {len(recs)} recommendations based on your {len(st.session_state.selected_anime)} selected anime!")
            
            # Display recommendations
            for i, rec_idx in enumerate(recs, 1):
                with st.expander(f"#{i} - {df.loc[rec_idx, 'title']}", expanded=(i<=5)):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Score:** {df.loc[rec_idx, 'Score']:.2f}/10")
                        st.write(f"**Type:** {df.loc[rec_idx, 'Type']}")
                        st.write(f"**Episodes:** {df.loc[rec_idx, 'Episodes']}")
                        st.write(f"**Members:** {df.loc[rec_idx, 'Members']:,.0f}")
                        st.write(f"**Genres:** {', '.join(df.loc[rec_idx, 'genres_list'])}")
                    
                    with col2:
                        st.write("**Matches with your selection:**")
                        
                        # Show overlap with each selected anime
                        for sel_idx in st.session_state.selected_anime:
                            sel_genres = set(df.loc[sel_idx, 'genres_list'])
                            rec_genres = set(df.loc[rec_idx, 'genres_list'])
                            overlap = sel_genres & rec_genres
                            
                            if overlap:
                                st.write(f"• {df.loc[sel_idx, 'title'][:30]}: {', '.join(list(overlap)[:2])}")
    else:
        st.info("👆 Select at least 2 anime to generate recommendations")
else:
    st.info("👆 Search and add anime to your profile to get started!")
    
    st.subheader("💡 Suggestion: Try These Combinations")
    
    st.markdown("""
    **Action Lovers:**
    - Attack on Titan + Fullmetal Alchemist: Brotherhood
    
    **Mystery Fans:**
    - Death Note + Steins;Gate
    
    **Slice of Life:**
    - Your Lie in April + Clannad
    
    **Diverse Tastes:**
    - One Punch Man + Death Parade + Mob Psycho 100
    """)
