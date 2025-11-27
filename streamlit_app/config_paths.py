import os
from pathlib import Path

# Get the directory where this file is located
APP_DIR = Path(__file__).parent

# Data file paths
ANIME_FEATURES = APP_DIR / "anime_features.parquet"
EMBEDDINGS_COMBINED = APP_DIR / "embeddings_combined.npy"
GRAPH_FEATURES = APP_DIR / "graph_features.parquet"
FAISS_INDEX = APP_DIR / "faiss_index_ivf.bin"
LGBM_MODEL = APP_DIR / "lgbm_ranker.txt"
LGBM_CONFIG = APP_DIR / "lgbm_config.json"
EVALUATION_RESULTS = APP_DIR / "evaluation_results.json"
