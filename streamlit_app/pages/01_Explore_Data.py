import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Explore Data", page_icon="📊", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_parquet(CURRENT_DIR / 'anime_features.parquet')

df = load_data()

st.title("📊 Explore Anime Dataset")

# Dataset overview
st.header("Dataset Overview")

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

# Filters
st.sidebar.header("Filters")

# Score filter
score_range = st.sidebar.slider(
    "Score Range",
    float(df['Score'].min()),
    float(df['Score'].max()),
    (6.0, 10.0)
)

# Type filter
anime_types = ['All'] + sorted(df['Type'].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Anime Type", anime_types)

# Apply filters
filtered_df = df[(df['Score'] >= score_range[0]) & (df['Score'] <= score_range[1])]
if selected_type != 'All':
    filtered_df = filtered_df[filtered_df['Type'] == selected_type]

st.info(f"Showing {len(filtered_df):,} anime")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🎭 Genres", "🏢 Studios", "📋 Browse"])

with tab1:
    st.subheader("Score Distribution")
    
    fig = px.histogram(
        filtered_df, 
        x='Score', 
        nbins=30,
        title="Anime Score Distribution",
        labels={'Score': 'Score', 'count': 'Number of Anime'}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Members Distribution")
        fig = px.histogram(
            filtered_df[filtered_df['Members'] > 0],
            x='log_members',
            nbins=30,
            title="Log(Members) Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Episodes Distribution")
        fig = px.histogram(
            filtered_df[filtered_df['Episodes'] <= 100],
            x='Episodes',
            nbins=30,
            title="Episodes Distribution (<100)"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Genre Analysis")
    
    # Genre counts
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    genre_counts = {col.replace('genre_', ''): df[col].sum() for col in genre_cols}
    genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
    genre_df = genre_df.sort_values('Count', ascending=True).tail(15)
    
    fig = px.bar(
        genre_df,
        x='Count',
        y='Genre',
        orientation='h',
        title="Top 15 Genres by Anime Count"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Genre scores
    st.subheader("Average Score by Genre")
    genre_scores = {}
    for col in genre_cols:
        genre_name = col.replace('genre_', '')
        mask = df[col] == 1
        if mask.sum() > 0:
            genre_scores[genre_name] = df[mask]['Score'].mean()
    
    score_df = pd.DataFrame(list(genre_scores.items()), columns=['Genre', 'Avg Score'])
    score_df = score_df.sort_values('Avg Score', ascending=False).head(10)
    
    fig = px.bar(
        score_df,
        x='Avg Score',
        y='Genre',
        orientation='h',
        title="Top 10 Highest-Rated Genres"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Studio Analysis")
    
    # Top studios by count
    top_studios = df['primary_studio'].value_counts().head(15)
    
    fig = px.bar(
        x=top_studios.values,
        y=top_studios.index,
        orientation='h',
        title="Top 15 Studios by Anime Count",
        labels={'x': 'Number of Anime', 'y': 'Studio'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Studio quality
    st.subheader("Studio Quality (Min 10 Anime)")
    studio_scores = df[df['primary_studio'] != 'Unknown'].groupby('primary_studio').agg({
        'Score': 'mean',
        'title': 'count'
    }).rename(columns={'title': 'count'})
    studio_scores = studio_scores[studio_scores['count'] >= 10].sort_values('Score', ascending=False).head(10)
    
    fig = px.bar(
        studio_scores.reset_index(),
        x='Score',
        y='primary_studio',
        orientation='h',
        title="Top 10 Studios by Average Score (Min 10 Anime)",
        labels={'Score': 'Average Score', 'primary_studio': 'Studio'}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Browse Anime")
    
    # Sorting
    sort_by = st.selectbox(
        "Sort by",
        ['Score', 'Members', 'Favorites', 'Popularity', 'title']
    )
    
    sort_ascending = st.checkbox("Ascending", value=False)
    
    # Display
    display_df = filtered_df.sort_values(sort_by, ascending=sort_ascending)
    
    # Select columns to display
    display_cols = ['title', 'Score', 'Type', 'Episodes', 'Members', 'Genres']
    
    st.dataframe(
        display_df[display_cols].head(100),
        use_container_width=True,
        height=600
    )
    
    st.caption(f"Showing top 100 of {len(display_df):,} anime")
