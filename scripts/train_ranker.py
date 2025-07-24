# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
# import logging
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# import json
# import numpy as np
# from pathlib import Path
# from app.relevance_model import RelevanceModel
# from utils.config_loader import load_config
# from utils.text_utils import get_keyword_frequency

# def extract_features_and_labels(chunk_data):
#     """
#     Extract features and synthetic labels from chunk data.
#     Args:
#         chunk_data (list): List of chunk dictionaries or lists from chunk_mapping.json.
#     Returns:
#         X (list): List of feature dictionaries.
#         y (list): List of relevance scores (synthetic).
#     """
#     X = []
#     y = []
#     for chunk in chunk_data:
#         try:
#             # Handle dictionary or list format
#             if isinstance(chunk, dict):
#                 text = chunk["text"]
#                 file_name = chunk["file"]
#                 start = chunk["start"]
#                 metadata = chunk.get("metadata", {})
#             elif isinstance(chunk, list) and len(chunk) >= 4:
#                 file_name, start, _, text = chunk[:4]
#                 metadata = {}  # No metadata in list format
#             else:
#                 logging.error(f"Invalid chunk format: {chunk}")
#                 continue

#             keyword_freq = get_keyword_frequency(text)
#             features = {
#                 "keyword_freq": sum(keyword_freq.values()) or 0,
#                 "position_weight": 1.0 / (start + 1),  # Earlier chunks are more relevant
#                 "section_weight": metadata.get("section_weight", 1.0),
#                 "chunk_length": len(text)
#             }
#             # Synthetic label: weighted combination of features
#             label = features["position_weight"] * 0.5 + features["keyword_freq"] * 0.3 + features["section_weight"] * 0.2
#             X.append(features)
#             y.append(label)
#             logging.debug(f"Extracted features for chunk from {file_name}: {features}, label: {label}")
#         except Exception as e:
#             logging.error(f"Error processing chunk from {file_name if 'file_name' in locals() else 'unknown'}: {e}")
#             continue
#     return X, y

# def main():
#     try:
#         config = load_config('config.yaml')
#         chunk_mapping_path = os.path.join(os.path.dirname(config['faiss']['index_path']), "chunk_mapping.json")
#         model_path = config['relevance_model']['path']

#         # Load chunk data
#         if not os.path.exists(chunk_mapping_path):
#             logging.error(f"Chunk mapping not found at {chunk_mapping_path}")
#             raise FileNotFoundError(f"Chunk mapping not found at {chunk_mapping_path}")
        
#         with open(chunk_mapping_path, 'r') as f:
#             chunk_mapping = json.load(f)
#             if isinstance(chunk_mapping, dict) and "chunks" in chunk_mapping:
#                 chunk_data = chunk_mapping["chunks"]
#             else:
#                 chunk_data = chunk_mapping  # Handle old list-based format
#             if not chunk_data:
#                 logging.error("No chunks found in chunk_mapping.json")
#                 raise ValueError("No chunks found in chunk_mapping.json")
#             logging.debug(f"Loaded {len(chunk_data)} chunks from {chunk_mapping_path}")

#         # Extract features and labels
#         X, y = extract_features_and_labels(chunk_data)
#         if not X or not y:
#             logging.error("No valid features or labels extracted")
#             raise ValueError("No valid features or labels extracted")
#         logging.debug(f"Extracted {len(X)} feature sets and labels")

#         # Train relevance model
#         relevance_model = RelevanceModel()
#         relevance_model.train(X, y)
#         relevance_model.save(model_path)
#         logging.info(f"Relevance model trained and saved to {model_path}")
#     except Exception as e:
#         logging.error(f"Failed to train relevance model: {e}")
#         raise

# if __name__ == "__main__":
#     main()