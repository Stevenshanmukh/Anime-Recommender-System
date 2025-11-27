import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Multi-Title Recommend", page_icon="🎭", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet(CURRENT_DIR / 'anime_features.parquet')
    return df

df = load_data()

st.title("🎭 Multi-Title Recommendations")
st.markdown("Combine multiple anime to get recommendations that match your diverse tastes!")

st.info("🚧 This feature is under construction. Check back soon!")

st.subheader("Top Rated Anime")
top_rated = df.nlargest(20, 'Score')[['title', 'Score', 'Type', 'Genres']]
st.dataframe(top_rated, use_container_width=True)
