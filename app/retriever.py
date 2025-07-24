# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
# import logging
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# from sentence_transformers import SentenceTransformer
# import faiss
# import json
# import numpy as np

# class Retriever:
#     def __init__(self, model_name, index_path, relevance_model_path=None):
#         logging.debug(f"Input model name: {model_name}")
#         model_path = (
#             "/Users/nimishgupta/Documents/rag_doc/models/all-MiniLM-L6-v2"
#             if model_name == "all-MiniLM-L6-v2"
#             else model_name
#         )
#         logging.debug(f"Loading model from: {model_path}")
#         try:
#             self.model = SentenceTransformer(model_path, local_files_only=True)
#             logging.debug("Model loaded successfully")
#         except Exception as e:
#             logging.error(f"Failed to load model: {e}")
#             raise
#         logging.debug(f"Loading FAISS index: {index_path}")
#         try:
#             self.index = faiss.read_index(index_path)
#         except Exception as e:
#             logging.error(f"Failed to load FAISS index: {e}")
#             raise
#         chunk_mapping_path = os.path.join(os.path.dirname(index_path), "chunk_mapping.json")
#         logging.debug(f"Loading chunk mapping: {chunk_mapping_path}")
#         try:
#             with open(chunk_mapping_path, "r") as f:
#                 loaded_data = json.load(f)
#                 # Handle list or dictionary format
#                 if isinstance(loaded_data, list):
#                     logging.warning(f"Converting list-based chunk_mapping.json to dictionary format")
#                     converted_chunks = []
#                     for chunk in loaded_data:
#                         if isinstance(chunk, list) and len(chunk) >= 4:
#                             converted_chunks.append({
#                                 "file": chunk[0],
#                                 "start": chunk[1],
#                                 "end": chunk[2],
#                                 "text": chunk[3]
#                             })
#                         else:
#                             logging.warning(f"Skipping invalid chunk entry: {chunk}")
#                     self.chunk_mapping = {"chunks": converted_chunks}
#                     # Save converted format
#                     with open(chunk_mapping_path, "w") as f:
#                         json.dump(self.chunk_mapping, f)
#                 else:
#                     self.chunk_mapping = loaded_data
#         except Exception as e:
#             logging.error(f"Failed to load chunk mapping: {e}")
#             raise

#     def retrieve(self, query, top_k=5):
#         """
#         Retrieve top-k relevant chunks for a query.
#         Args:
#             query (str): The query string.
#             top_k (int): Number of chunks to retrieve.
#         Returns:
#             list: List of chunk texts.
#         """
#         try:
#             query_embedding = self.model.encode([query], show_progress_bar=False)[0]
#             query_embedding = np.array([query_embedding]).astype('float32')
#             distances, indices = self.index.search(query_embedding, top_k)
#             chunks = []
#             for i in indices[0]:
#                 if i < len(self.chunk_mapping["chunks"]):
#                     chunk = self.chunk_mapping["chunks"][i]
#                     if isinstance(chunk, dict):
#                         chunks.append(chunk["text"])
#                     else:
#                         logging.warning(f"Skipping invalid chunk at index {i}: {chunk}")
#             logging.debug(f"Retrieved {len(chunks)} chunks for query: {query}")
#             return chunks
#         except Exception as e:
#             logging.error(f"Error retrieving chunks: {e}")
#             return []

# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
# import logging
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import json
# from pathlib import Path
# import pickle
# from utils.text_utils import extract_keywords
# from app.relevance_model import RelevanceModel

# class Retriever:
#     def __init__(self, model_name, index_path, relevance_model_path):
#         logging.debug(f"Initializing Retriever with model: {model_name}, index_path: {index_path}, relevance_model_path: {relevance_model_path}")
#         try:
#             self.model = SentenceTransformer(model_name, local_files_only=True)
#             logging.debug("SentenceTransformer loaded successfully")
#         except Exception as e:
#             logging.error(f"Failed to load SentenceTransformer: {e}")
#             raise
        
#         self.index_path = index_path
#         try:
#             self.index = faiss.read_index(index_path)
#             logging.debug(f"FAISS index loaded from {index_path}")
#         except Exception as e:
#             logging.error(f"Failed to load FAISS index: {e}")
#             raise
        
#         chunk_mapping_path = Path(index_path).parent / "chunk_mapping.json"
#         try:
#             with open(chunk_mapping_path, "r") as f:
#                 self.chunk_mapping = json.load(f)
#                 if isinstance(self.chunk_mapping, list):
#                     logging.warning(f"Converting list-based chunk_mapping.json to dictionary format")
#                     self.chunk_mapping = {"chunks": [
#                         {
#                             "file": chunk[0],
#                             "start": chunk[1],
#                             "end": chunk[2],
#                             "text": chunk[3]
#                         } for chunk in self.chunk_mapping if isinstance(chunk, list) and len(chunk) >= 4
#                     ]}
#             logging.debug(f"Loaded {len(self.chunk_mapping.get('chunks', []))} chunks from {chunk_mapping_path}")
#         except Exception as e:
#             logging.error(f"Failed to load chunk mapping: {e}")
#             raise
        
#         self.relevance_model = None
#         if relevance_model_path and os.path.exists(relevance_model_path):
#             try:
#                 self.relevance_model = RelevanceModel()
#                 self.relevance_model.load(relevance_model_path)
#                 logging.debug(f"Relevance model loaded from {relevance_model_path}")
#             except Exception as e:
#                 logging.error(f"Failed to load relevance model: {e}")
#                 self.relevance_model = None

#     def retrieve(self, query, top_k=5):
#         """
#         Retrieve top-k relevant chunks for a query using relevance model and FAISS.
#         Args:
#             query (str): User query.
#             top_k (int): Number of chunks to return.
#         Returns:
#             list: List of chunk texts.
#         """
#         try:
#             # Embed query
#             query_embedding = self.model.encode([query], show_progress_bar=False)[0].astype('float32')
#             logging.debug(f"Encoded query: {query}")

#             # Get candidate chunks using relevance model
#             candidate_indices = list(range(self.index.ntotal))
#             if self.relevance_model:
#                 try:
#                     query_keywords = extract_keywords(query)
#                     X = []
#                     for i in candidate_indices:
#                         chunk = self.chunk_mapping["chunks"][i]
#                         keywords = extract_keywords(chunk["text"])
#                         features = {
#                             "keyword_freq": sum(chunk.get("metadata", {}).get("keyword_freq", {}).values()) or len(keywords),
#                             "position_weight": 1.0 / (chunk["start"] + 1),
#                             "section_weight": chunk.get("metadata", {}).get("section_weight", 1.0),
#                             "chunk_length": len(chunk["text"])
#                         }
#                         X.append(features)
#                     scores = self.relevance_model.predict(X)
#                     # Select top 50% of chunks by relevance score
#                     candidate_indices = [i for _, i in sorted(zip(scores, candidate_indices), reverse=True)][:int(len(candidate_indices) * 0.5)]
#                     logging.debug(f"Selected {len(candidate_indices)} candidate chunks using relevance model")
#                 except Exception as e:
#                     logging.error(f"Relevance model prediction failed: {e}")
#                     candidate_indices = list(range(self.index.ntotal))  # Fallback to all chunks

#             # FAISS search on candidates
#             if not candidate_indices:
#                 logging.warning("No candidate chunks selected, returning empty list")
#                 return []
            
#             # Extract embeddings for candidates
#             candidate_embeddings = np.zeros((len(candidate_indices), self.index.d), dtype='float32')
#             for j, idx in enumerate(candidate_indices):
#                 candidate_embeddings[j] = self.index.reconstruct(idx)
            
#             # Search
#             distances, indices = self.index.search(np.array([query_embedding]), top_k)
#             retrieved_chunks = []
#             for idx in indices[0]:
#                 if idx in candidate_indices:
#                     chunk = self.chunk_mapping["chunks"][idx]["text"]
#                     retrieved_chunks.append(chunk)
#             logging.debug(f"Retrieved {len(retrieved_chunks)} chunks for query")
#             return retrieved_chunks[:top_k]
#         except Exception as e:
#             logging.error(f"Retrieval failed: {e}")
#             return []


# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
# import logging
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import json
# from pathlib import Path
# from utils.text_utils import get_keyword_frequency
# from app.relevance_model import RelevanceModel

# class Retriever:
#     def __init__(self, model_name, index_path, relevance_model_path=None):
#         logging.debug(f"Initializing Retriever with model: {model_name}, index_path: {index_path}, relevance_model_path: {relevance_model_path}")
#         try:
#             self.model = SentenceTransformer(model_name, local_files_only=True)
#             logging.debug("SentenceTransformer loaded successfully")
#         except Exception as e:
#             logging.error(f"Failed to load SentenceTransformer: {e}")
#             raise
        
#         self.index_path = index_path
#         try:
#             self.index = faiss.read_index(index_path)
#             logging.debug(f"FAISS index loaded from {index_path}")
#         except Exception as e:
#             logging.error(f"Failed to load FAISS index: {e}")
#             raise
        
#         chunk_mapping_path = Path(index_path).parent / "chunk_mapping.json"
#         try:
#             with open(chunk_mapping_path, "r") as f:
#                 self.chunk_mapping = json.load(f)
#                 if isinstance(self.chunk_mapping, list):
#                     logging.warning(f"Converting list-based chunk_mapping.json to dictionary format")
#                     self.chunk_mapping = {"chunks": [
#                         {
#                             "file": chunk[0],
#                             "start": chunk[1],
#                             "end": chunk[2],
#                             "text": chunk[3],
#                             "metadata": {}
#                         } for chunk in self.chunk_mapping if isinstance(chunk, list) and len(chunk) >= 4
#                     ]}
#             logging.debug(f"Loaded {len(self.chunk_mapping.get('chunks', []))} chunks from {chunk_mapping_path}")
#         except Exception as e:
#             logging.error(f"Failed to load chunk mapping: {e}")
#             raise
        
#         self.relevance_model = None
#         if relevance_model_path and os.path.exists(relevance_model_path):
#             try:
#                 self.relevance_model = RelevanceModel()
#                 self.relevance_model.load(relevance_model_path)
#                 logging.debug(f"Relevance model loaded from {relevance_model_path}")
#             except Exception as e:
#                 logging.error(f"Failed to load relevance model: {e}")
#                 self.relevance_model = None

#     def retrieve(self, query, top_k=5):
#         """
#         Retrieve top-k relevant chunks for a query using relevance model and FAISS.
#         Args:
#             query (str): User query.
#             top_k (int): Number of chunks to return.
#         Returns:
#             list: List of chunk texts.
#         """
#         try:
#             # Embed query
#             query_embedding = self.model.encode([query], show_progress_bar=False)[0].astype('float32')
#             logging.debug(f"Encoded query: {query}")

#             # Get candidate chunks using relevance model
#             candidate_indices = list(range(self.index.ntotal))
#             if self.relevance_model:
#                 try:
#                     X = []
#                     for i in candidate_indices:
#                         chunk = self.chunk_mapping["chunks"][i]
#                         keyword_freq = get_keyword_frequency(chunk["text"])
#                         features = {
#                             "keyword_freq": sum(keyword_freq.values()) or 0,
#                             "position_weight": 1.0 / (chunk["start"] + 1),
#                             "section_weight": chunk.get("metadata", {}).get("section_weight", 1.0),
#                             "chunk_length": len(chunk["text"])
#                         }
#                         X.append(features)
#                     scores = self.relevance_model.predict(X)
#                     # Select top 50% of chunks by relevance score
#                     candidate_indices = [i for _, i in sorted(zip(scores, candidate_indices), reverse=True)][:int(len(candidate_indices) * 0.5)]
#                     logging.debug(f"Selected {len(candidate_indices)} candidate chunks using relevance model")
#                 except Exception as e:
#                     logging.error(f"Relevance model prediction failed: {e}")
#                     candidate_indices = list(range(self.index.ntotal))  # Fallback to all chunks

#             # FAISS search on candidates
#             if not candidate_indices:
#                 logging.warning("No candidate chunks selected, returning empty list")
#                 return []
            
#             # Extract embeddings for candidates
#             candidate_embeddings = np.zeros((len(candidate_indices), self.index.d), dtype='float32')
#             for j, idx in enumerate(candidate_indices):
#                 candidate_embeddings[j] = self.index.reconstruct(idx)
            
#             # Search
#             distances, indices = self.index.search(np.array([query_embedding]), top_k)
#             retrieved_chunks = []
#             for idx in indices[0]:
#                 if idx in candidate_indices:
#                     chunk = self.chunk_mapping["chunks"][idx]["text"]
#                     retrieved_chunks.append(chunk)
#             logging.debug(f"Retrieved {len(retrieved_chunks)} chunks for query")
#             return retrieved_chunks[:top_k]
#         except Exception as e:
#             logging.error(f"Retrieval failed: {e}")
#             return []


import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path
from utils.text_utils import get_keyword_frequency, extract_keywords
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, model_name, index_path, metadata_dir):
        logging.debug(f"Initializing Retriever with model: {model_name}, index_path: {index_path}")
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
            logging.debug("SentenceTransformer loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer: {e}")
            raise
        
        self.index_path = index_path
        try:
            self.index = faiss.read_index(index_path)
            logging.debug(f"FAISS index loaded from {index_path}")
        except Exception as e:
            logging.error(f"Failed to load FAISS index: {e}")
            raise
        
        self.metadata_dir = metadata_dir
        chunk_mapping_path = Path(index_path).parent / "chunk_mapping.json"
        try:
            with open(chunk_mapping_path, "r") as f:
                self.chunk_mapping = json.load(f)
                if isinstance(self.chunk_mapping, list):
                    logging.warning("Converting list-based chunk_mapping.json to dictionary format")
                    self.chunk_mapping = {"chunks": [
                        {
                            "file": chunk[0],
                            "start": chunk[1],
                            "end": chunk[2],
                            "text": chunk[3],
                            "metadata": {}
                        } for chunk in self.chunk_mapping if isinstance(chunk, list) and len(chunk) >= 4
                    ]}
            logging.debug(f"Loaded {len(self.chunk_mapping.get('chunks', []))} chunks from {chunk_mapping_path}")
        except Exception as e:
            logging.error(f"Failed to load chunk mapping: {e}")
            raise

    def retrieve(self, query, top_k=5, top_docs=3, min_doc_score=0.1):
        """
        Retrieve top-k chunks from top documents based on query similarity.
        Args:
            query (str): User query.
            top_k (int): Number of chunks to return.
            top_docs (int): Number of documents to select.
            min_doc_score (float): Minimum similarity score for documents.
        Returns:
            list: List of chunk texts.
        """
        try:
            # Embed query
            query_embedding = self.model.encode([query], show_progress_bar=False)[0].astype('float32')
            query_keywords = extract_keywords(query, top_n=10)
            logging.debug(f"Encoded query: {query}, keywords: {query_keywords}")

            # Get document-level similarity
            doc_scores = {}
            doc_chunks = {}
            for i, chunk in enumerate(self.chunk_mapping["chunks"]):
                file_name = chunk["file"]
                chunk_embedding = self.index.reconstruct(i)
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                
                if file_name not in doc_scores:
                    doc_scores[file_name] = []
                    doc_chunks[file_name] = []
                doc_scores[file_name].append(similarity)
                doc_chunks[file_name].append((i, chunk, similarity))

            # Aggregate document scores
            for file_name in doc_scores:
                doc_scores[file_name] = max(doc_scores[file_name])  # Use max chunk similarity

            # Select top documents
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_docs]
            selected_chunks = []
            for file_name, score in sorted_docs:
                if score < min_doc_score:
                    continue
                for i, chunk, similarity in doc_chunks[file_name]:
                    section_weight = chunk.get("metadata", {}).get("section_weight", 1.0)
                    keyword_freq = chunk.get("metadata", {}).get("keyword_freq", {})
                    keyword_score = sum(keyword_freq.get(kw, 0) for kw in query_keywords)
                    final_score = similarity * section_weight + keyword_score * 0.1
                    selected_chunks.append((final_score, chunk["text"]))

            # Return top-k chunks
            selected_chunks = sorted(selected_chunks, key=lambda x: x[0], reverse=True)[:top_k]
            return [chunk for _, chunk in selected_chunks]
        except Exception as e:
            logging.error(f"Error retrieving chunks for query '{query}': {str(e)}")
            return []