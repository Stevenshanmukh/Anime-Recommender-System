import streamlit as st
import requests
from pathlib import Path
import tarfile
import os

@st.cache_resource
def download_model_files():
    """Download large model files from GitHub Releases if not present"""
    
    files_needed = ['embeddings_combined.npy', 'faiss_index_ivf.bin']
    all_exist = all(Path(f).exists() for f in files_needed)
    
    if all_exist:
        return True
    
    st.info("⏳ First-time setup: Downloading model files (555MB)...")
    st.warning("This will take 2-3 minutes. Please wait...")
    
    # GitHub Release URL (update after creating release)
    release_url = "https://github.com/Stevenshanmukh/Anime-Recommender-System/releases/download/v1.0.0/model_files.tar.gz"
    
    try:
        # Download
        response = requests.get(release_url, stream=True)
        response.raise_for_status()
        
        tar_path = Path("model_files.tar.gz")
        
        # Save with progress
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0)
        
        with open(tar_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    progress_bar.progress(downloaded / total_size)
        
        # Extract
        st.info("📦 Extracting files...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall()
        
        # Cleanup
        tar_path.unlink()
        
        st.success("✅ Model files ready!")
        return True
        
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        st.info("Please download manually from GitHub Releases")
        return False
