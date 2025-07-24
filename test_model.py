import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
import logging
logging.basicConfig(level=logging.DEBUG)
from sentence_transformers import SentenceTransformer
try:
model = SentenceTransformer(
"/Users/nimishgupta/Documents/rag_doc/models/all-MiniLM-L6-v2",
local_files_only=True
)
print("Model loaded successfully!")
except Exception as e:
print(f"Error: {e}")
