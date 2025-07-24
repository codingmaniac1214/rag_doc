import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class Embedder:
    def __init__(self, model_name, index_path):
        logging.debug(f"Initializing Embedder with model: {model_name}, index_path: {index_path}")
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
            logging.debug("SentenceTransformer loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer: {e}")
            raise
        self.index_path = index_path
        self.index = None
        self.chunk_mapping = {"chunks": []}
    
    def create_index(self, chunk_files):
        """
        Generate embeddings and build FAISS index from chunk files.
        Args:
            chunk_files (list): List of paths to chunk JSON files.
        """
        try:
            texts = []
            new_chunks = []
            chunk_mapping_path = os.path.join(os.path.dirname(self.index_path), 'chunk_mapping.json')

            # Load existing FAISS index and chunk mapping
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    logging.debug(f"Loaded existing FAISS index from {self.index_path}")
                except Exception as e:
                    logging.error(f"Failed to load FAISS index: {e}")
                    raise
            if os.path.exists(chunk_mapping_path):
                try:
                    with open(chunk_mapping_path, "r") as f:
                        self.chunk_mapping = json.load(f)
                        if isinstance(self.chunk_mapping, list):
                            logging.warning(f"Converting list-based chunk_mapping.json to dictionary format")
                            self.chunk_mapping = {"chunks": [
                                {
                                    "file": chunk[0],
                                    "start": chunk[1],
                                    "end": chunk[2],
                                    "text": chunk[3]
                                } for chunk in self.chunk_mapping if isinstance(chunk, list) and len(chunk) >= 4
                            ]}
                    logging.debug(f"Loaded {len(self.chunk_mapping['chunks'])} chunks from {chunk_mapping_path}")
                except Exception as e:
                    logging.error(f"Failed to load chunk mapping: {e}")
                    raise

            # Process chunk files
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if not isinstance(data, dict) or "chunks" not in data:
                            logging.warning(f"Invalid chunk file format in {chunk_file}, expected {'chunks': [...]}, skipping")
                            continue
                        chunks = data["chunks"]
                        base_name = os.path.splitext(os.path.basename(chunk_file))[0].replace('_chunks', '')
                        file_path = os.path.join('data/raw_pdfs', base_name + '.pdf')
                        if not os.path.exists(file_path):
                            file_path = os.path.join('data/raw_pdfs', base_name + '.txt')
                        if str(file_path) in {chunk["file"] for chunk in self.chunk_mapping["chunks"] if isinstance(chunk, dict)}:
                            logging.debug(f"Skipping already indexed file: {file_path}")
                            continue
                        texts.extend(chunks)
                        for i, chunk in enumerate(chunks):
                            new_chunks.append({
                                "file": file_path,
                                "start": i * 500,  # Approximate, assuming chunk_size=1000, overlap=200
                                "end": i * 500 + len(chunk),
                                "text": chunk
                            })
                    logging.debug(f"Processed chunk file: {chunk_file}")
                except Exception as e:
                    logging.error(f"Error processing chunk file {chunk_file}: {str(e)}")
                    continue
            
            if not texts:
                logging.info("No new chunks to process.")
                return

            # Generate embeddings
            try:
                embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
                embeddings = np.array(embeddings).astype('float32')
                logging.debug(f"Generated {len(embeddings)} embeddings")
            except Exception as e:
                logging.error(f"Error generating embeddings: {str(e)}")
                raise
            
            # Initialize or update FAISS index
            try:
                if self.index is None:
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    logging.debug(f"Initialized new FAISS index with dimension {dimension}")
                self.index.add(embeddings)
                self.chunk_mapping["chunks"].extend(new_chunks)
                logging.debug(f"Added {len(embeddings)} embeddings to FAISS index")
            except Exception as e:
                logging.error(f"Error adding to FAISS index: {str(e)}")
                raise
            
            # Save index and mapping
            try:
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                faiss.write_index(self.index, self.index_path)
                logging.debug(f"Saved FAISS index to {self.index_path}")
                with open(chunk_mapping_path, 'w') as f:
                    json.dump(self.chunk_mapping, f)
                logging.debug(f"Saved chunk mapping to {chunk_mapping_path}")
                logging.info(f"FAISS index updated with {len(texts)} new chunks and saved to {self.index_path}")
            except Exception as e:
                logging.error(f"Error saving FAISS index or chunk mapping: {str(e)}")
                raise
        
        except Exception as e:
            logging.error(f"Failed to create FAISS index: {str(e)}")
            raise