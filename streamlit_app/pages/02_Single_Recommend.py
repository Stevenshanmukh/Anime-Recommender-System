import streamlit as st
import pandas as pd
import numpy as np
import faiss
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Single Recommend", page_icon="🎯", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_parquet('anime_features.parquet')
    embeddings = np.load('embeddings_combined.npy')
    graph_features = pd.read_parquet('graph_features.parquet')
    return df, embeddings, graph_features

@st.cache_resource
def load_models():
    index = faiss.read_index('faiss_index_ivf.bin')
    model = lgb.Booster(model_file='lgbm_ranker.txt')
    return index, model

df, embeddings, graph_features = load_data()
faiss_index, lgbm_model = load_models()

# Feature extraction function
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
    
    features = [
        cosine_sim, *cand_graph, cand_score, cand_members, cand_favorites,
        cand_popularity, genre_overlap, genre_jaccard, studio_overlap,
        has_score, is_highly_rated
    ]
    
    return np.array(features)

def get_recommendations(query_idx, k=10):
    query_emb = embeddings[query_idx].reshape(1, -1)
    n_candidates = k * 5
    distances, indices = faiss_index.search(query_emb, n_candidates + 1)
    
    mask = indices[0] != query_idx
    candidates = indices[0][mask][:n_candidates]
    
    candidate_features = np.array([
        create_ranking_features(query_idx, cand_idx)
        for cand_idx in candidates
    ])
    
    scores = lgbm_model.predict(candidate_features)
    top_k_idx = np.argsort(scores)[::-1][:k]
    
    return candidates[top_k_idx], scores[top_k_idx]

def explain_recommendation(query_idx, rec_idx):
    explanations = []
    
    rec_score = df.loc[rec_idx, 'Score']
    if pd.notna(rec_score):
        if rec_score >= 8.5:
            explanations.append(f"⭐ Exceptionally highly rated ({rec_score:.2f}/10)")
        elif rec_score >= 8.0:
            explanations.append(f"⭐ Highly rated ({rec_score:.2f}/10)")
        elif rec_score >= 7.0:
            explanations.append(f"✓ Well-rated ({rec_score:.2f}/10)")
    
    # Genre overlap
    query_genres = set(df.loc[query_idx, 'genres_list'])
    rec_genres = set(df.loc[rec_idx, 'genres_list'])
    shared = query_genres & rec_genres
    if len(shared) >= 2:
        explanations.append(f"🎭 Shares genres: {', '.join(list(shared)[:2])}")
    elif len(shared) == 1:
        explanations.append(f"🎭 Shares genre: {list(shared)[0]}")
    
    # Studio overlap
    query_studios = set(df.loc[query_idx, 'studios_list'])
    rec_studios = set(df.loc[rec_idx, 'studios_list'])
    shared_studios = query_studios & rec_studios
    if shared_studios and 'Unknown' not in shared_studios:
        explanations.append(f"🎬 Same studio: {list(shared_studios)[0]}")
    
    # Popularity
    members = df.loc[rec_idx, 'Members']
    if members > 1000000:
        explanations.append(f"👥 Very popular ({members:,.0f} members)")
    
    return explanations[:4]

# UI
st.title("🎯 Single Anime Recommendations")

st.markdown("""
Get personalized recommendations based on a single anime you love!
The system uses multi-modal deep learning to find similar anime.
""")

# Search
st.subheader("Select an Anime")

search_query = st.text_input("🔍 Search by title", placeholder="e.g., Death Note, Naruto...")

if search_query:
    matches = df[df['title'].str.contains(search_query, case=False, na=False)]
    
    if len(matches) > 0:
        anime_options = {f"{row['title']} ({row['Type']}, {row['Score']:.2f})": idx 
                        for idx, row in matches.head(20).iterrows()}
        
        selected = st.selectbox("Select anime", list(anime_options.keys()))
        
        if selected:
            query_idx = anime_options[selected]
            
            # Display selected anime
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Selected Anime")
                st.write(f"**{df.loc[query_idx, 'title']}**")
                st.write(f"Score: {df.loc[query_idx, 'Score']:.2f}/10")
                st.write(f"Type: {df.loc[query_idx, 'Type']}")
                st.write(f"Episodes: {df.loc[query_idx, 'Episodes']}")
                st.write(f"Members: {df.loc[query_idx, 'Members']:,.0f}")
            
            with col2:
                st.subheader("Info")
                st.write(f"**Genres:** {', '.join(df.loc[query_idx, 'genres_list'])}")
                st.write(f"**Studios:** {', '.join(df.loc[query_idx, 'studios_list'][:3])}")
                if df.loc[query_idx, 'description']:
                    st.write(f"**Description:** {df.loc[query_idx, 'description'][:200]}...")
            
            # Get recommendations
            st.markdown("---")
            st.subheader("🎬 Recommended Anime")
            
            num_recs = st.slider("Number of recommendations", 5, 20, 10)
            
            with st.spinner("Generating recommendations..."):
                recs, scores = get_recommendations(query_idx, k=num_recs)
            
            # Display recommendations
            for i, rec_idx in enumerate(recs, 1):
                with st.expander(f"#{i} - {df.loc[rec_idx, 'title']}", expanded=(i<=3)):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"**Score:** {df.loc[rec_idx, 'Score']:.2f}/10")
                        st.write(f"**Type:** {df.loc[rec_idx, 'Type']}")
                        st.write(f"**Episodes:** {df.loc[rec_idx, 'Episodes']}")
                        st.write(f"**Members:** {df.loc[rec_idx, 'Members']:,.0f}")
                    
                    with col2:
                        st.write(f"**Genres:** {', '.join(df.loc[rec_idx, 'genres_list'])}")
                        
                        # Explanation
                        st.write("**Why recommended:**")
                        explanations = explain_recommendation(query_idx, rec_idx)
                        for exp in explanations:
                            st.write(f"• {exp}")
    else:
        st.warning("No anime found matching your search.")
else:
    st.info("👆 Enter an anime title to get started!")
    
    # Show popular anime as examples
    st.subheader("Popular Anime Examples")
    
    popular = df.nlargest(10, 'Members')[['title', 'Score', 'Type', 'Members']]
    st.dataframe(popular, use_container_width=True)
