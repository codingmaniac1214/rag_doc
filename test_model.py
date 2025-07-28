import os
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer

# Set offline mode and HF cache directory
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(Path("models").resolve())

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load model
try:
    model_path = Path("models/all-MiniLM-L6-v2").resolve()
    model = SentenceTransformer(str(model_path), local_files_only=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
