# Anime Recommender System - Streamlit App

🎬 **Production-Ready Multi-Modal Recommendation System**

## Features

- **Single Title Recommendations**: Get personalized recommendations based on one anime
- **Multi-Title Recommendations**: Combine multiple anime preferences
- **Data Exploration**: Interactive visualizations and statistics
- **Model Performance**: Comprehensive metrics and explainability

## System Specifications

- **Embeddings**: 3,647 dimensions (text + graph)
- **Search Speed**: 5ms average
- **Model**: LightGBM LambdaRank (NDCG = 1.0)
- **Quality**: 7.93/10 average recommendation score
- **Explainability**: SHAP-powered natural language explanations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install streamlit pandas numpy faiss-cpu lightgbm plotly scikit-learn
```

Or install all at once:
```bash
pip install streamlit pandas numpy faiss-cpu lightgbm plotly scikit-learn --break-system-packages
```

## Running the App

### Navigate to the app directory:
```bash
cd streamlit_app
```

### Run Streamlit:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Project Structure
```
streamlit_app/
├── app.py                          # Main landing page
├── pages/
│   ├── 01_Explore_Data.py         # Data exploration & visualization
│   ├── 02_Single_Recommend.py     # Single anime recommendations
│   ├── 03_Multi_Title_Recommend.py # Multi-anime recommendations
│   └── 04_Model_Card.py           # Model performance & documentation
├── assets/
│   └── styles.css                 # Custom CSS styles
└── [data files]
    ├── anime_features.parquet     # Anime dataset
    ├── embeddings_combined.npy    # Multi-modal embeddings
    ├── faiss_index_ivf.bin       # FAISS search index
    ├── lgbm_ranker.txt           # Trained ranking model
    ├── lgbm_config.json          # Model configuration
    ├── evaluation_results.json    # Performance metrics
    └── graph_features.parquet     # Graph network features
```

## Usage Guide

### 1. Home Page
- Overview of system capabilities
- Key metrics and features
- Navigation guide

### 2. Explore Data
- Browse 19,931 anime
- Interactive visualizations
- Filter by score, type, genre
- Studio and genre analysis

### 3. Single Recommendations
- Search for any anime
- Get top-K similar recommendations
- View detailed explanations
- Understand why each anime was recommended

### 4. Multi-Title Recommendations
- Build a profile with 2-5 anime
- Get recommendations matching your diverse tastes
- See overlap with each selected anime

### 5. Model Card
- Detailed performance metrics
- SHAP feature importance
- Model architecture documentation
- Comparison with baseline models

## Key Metrics

| Metric | Value |
|--------|-------|
| NDCG@10 | 1.0 (Perfect) |
| Avg Quality | 7.93/10 |
| Genre Relevance | 1.73 overlap |
| Search Speed | 5ms |
| Coverage | 345 anime |

## Technical Architecture

### Multi-Modal Embeddings
- **Text**: TF-IDF (word + char + title) = 3,500 dims
- **Graph**: SVD on studio-producer network = 74 dims
- **Total**: 3,574 dimensions

### Ranking Model
- **Algorithm**: LightGBM LambdaRank
- **Features**: 20 ranking features
- **Training**: 200 queries, 10K pairs
- **Convergence**: 3 iterations (fast!)

### Explainability
- SHAP values for feature importance
- Natural language explanations
- 6 explanation types (quality, content, studio, popularity)

## Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Data Files Not Found
Ensure you're running from the `streamlit_app` directory:
```bash
cd streamlit_app
streamlit run app.py
```

## Performance Notes

- First load may take 10-20 seconds (model loading)
- Subsequent operations are <100ms
- FAISS search: ~5ms
- Recommendation generation: ~50ms total

## Credits

Built with:
- **Streamlit** - Web framework
- **LightGBM** - Ranking model
- **FAISS** - Similarity search
- **Plotly** - Interactive visualizations
- **SHAP** - Model explainability

## License

This project is for educational and portfolio purposes.

---

**Enjoy exploring anime recommendations!** 🎬✨
