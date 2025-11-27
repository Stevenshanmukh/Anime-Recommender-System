import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Explore Data", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet(CURRENT_DIR / 'anime_features.parquet')

df = load_data()

st.title("📊 Explore Anime Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Anime", f"{len(df):,}")
with col2:
    st.metric("Avg Score", f"{df['Score'].mean():.2f}")
with col3:
    st.metric("Total Genres", f"{len([c for c in df.columns if c.startswith('genre_')])}")
with col4:
    st.metric("Total Studios", f"{df['primary_studio'].nunique():,}")

st.markdown("---")

# Simple data browser
st.subheader("Browse Anime")

sort_by = st.selectbox("Sort by", ['Score', 'Members', 'Favorites', 'title'])
display_df = df.sort_values(sort_by, ascending=False)

display_cols = ['title', 'Score', 'Type', 'Episodes', 'Members', 'Genres']
st.dataframe(display_df[display_cols].head(100), use_container_width=True, height=600)
