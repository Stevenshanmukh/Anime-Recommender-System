import streamlit as st
import requests
from pathlib import Path
import zipfile
import os

# Get current directory
CURRENT_DIR = Path(__file__).parent

@st.cache_resource
def download_model_files():
    """Download large model files from GitHub Releases if not present"""
    
    # Check in current directory
    files_needed = [
        CURRENT_DIR / 'embeddings_combined.npy',
        CURRENT_DIR / 'faiss_index_ivf.bin'
    ]
    
    all_exist = all(f.exists() for f in files_needed)
    
    if all_exist:
        return True
    
    st.info("⏳ First-time setup: Downloading model files (~200 MB compressed)...")
    st.warning("This will take 2-3 minutes on first run. Please wait...")
    
    # Your GitHub Release URL
    release_url = "https://github.com/Stevenshanmukh/Anime-Recommender-System/releases/download/v1.0.0/model_files.zip"
    
    try:
        # Download
        with st.spinner("Downloading model files..."):
            response = requests.get(release_url, stream=True, timeout=300)
            response.raise_for_status()
            
            zip_path = CURRENT_DIR / "model_files.zip"
            
            # Save file
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        # Extract to current directory
        with st.spinner("Extracting files..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(CURRENT_DIR)
        
        # Cleanup
        zip_path.unlink()
        
        st.success("✅ Model files downloaded and ready!")
        st.balloons()
        return True
        
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        st.info("""
        **Manual download:**
        1. Download from: https://github.com/Stevenshanmukh/Anime-Recommender-System/releases/tag/v1.0.0
        2. Extract model_files.zip to streamlit_app/
        3. Restart the app
        """)
        return False
