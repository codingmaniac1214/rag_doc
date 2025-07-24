# import os
# import json
# import fitz  # PyMuPDF
# from utils.text_utils import clean_text
# from utils.metadata_utils import extract_metadata

# def preprocess_pdf(pdf_path, output_dir, metadata_dir):
#     """
#     Extract text and metadata from a PDF using PyMuPDF and save to output directories.
#     """
#     try:
#         # Open PDF with PyMuPDF
#         doc = fitz.open(pdf_path)
        
#         # Extract text from all pages
#         raw_text = ""
#         for page in doc:
#             raw_text += page.get_text("text") + "\n"
#         cleaned_text = clean_text(raw_text)
        
#         # Extract metadata
#         pdf_metadata = doc.metadata
#         metadata = {
#             "source": os.path.basename(pdf_path),
#             "title": pdf_metadata.get("title", "Unknown Title"),
#             "author": pdf_metadata.get("author", "Unknown Author"),
#             "section_count": len(doc),  # Number of pages as a proxy for sections
#             "keyword_freq": extract_metadata(cleaned_text, pdf_path).get("keyword_freq", {}),
#             "position_weight": 0,
#             "section_weight": 0
#         }
        
#         # Save cleaned text
#         os.makedirs(output_dir, exist_ok=True)
#         base_name = os.path.splitext(os.path.basename(pdf_path))[0]
#         text_output_path = os.path.join(output_dir, f"{base_name}.txt")
#         with open(text_output_path, 'w', encoding='utf-8') as f:
#             f.write(cleaned_text)
        
#         # Save metadata
#         os.makedirs(metadata_dir, exist_ok=True)
#         metadata_output_path = os.path.join(metadata_dir, f"{base_name}.json")
#         with open(metadata_output_path, 'w', encoding='utf-8') as f:
#             json.dump(metadata, f, indent=4)
        
#         doc.close()
#         return text_output_path, metadata_output_path
#     except Exception as e:
#         print(f"Error processing {pdf_path}: {str(e)}")
#         return None, None

# def preprocess_directory(input_dir, output_dir, metadata_dir):
#     """
#     Process all PDFs in a directory.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(metadata_dir, exist_ok=True)
    
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.pdf'):
#             pdf_path = os.path.join(input_dir, filename)
#             preprocess_pdf(pdf_path, output_dir, metadata_dir)

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import faiss
import numpy as np
import json
from pathlib import Path
from itertools import chain
from utils.config_loader import load_config
from utils.text_utils import clean_text
from utils.metadata_utils import extract_metadata

# def preprocess_file(file_path, text_dir, metadata_dir, chunks_dir, chunk_size, chunk_overlap):
#     """
#     Extract text, metadata, and chunks from a PDF or text file, generate embeddings, and save to output directories.
#     Args:
#         file_path (str): Path to the file.
#         text_dir (str): Directory to save cleaned text.
#         metadata_dir (str): Directory to save metadata.
#         chunks_dir (str): Directory to save chunks.
#         chunk_size (int): Size of each text chunk.
#         chunk_overlap (int): Overlap between chunks.
#     Returns:
#         tuple: (text_output_path, metadata_output_path, chunk_output_path, chunks, embeddings)
#     """
#     try:
#         # Extract text
#         raw_text = ""
#         pdf_metadata = {}
#         if file_path.lower().endswith('.pdf'):
#             doc = fitz.open(file_path)
#             for page in doc:
#                 raw_text += page.get_text("text") + "\n"
#             pdf_metadata = doc.metadata
#         else:  # .txt
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 raw_text = f.read()

#         cleaned_text = clean_text(raw_text)
        
#         # Save cleaned text
#         os.makedirs(text_dir, exist_ok=True)
#         base_name = os.path.splitext(os.path.basename(file_path))[0]
#         text_output_path = os.path.join(text_dir, f"{base_name}.txt")
#         with open(text_output_path, 'w', encoding='utf-8') as f:
#             f.write(cleaned_text)
#         logging.debug(f"Saved cleaned text to {text_output_path}")

#         # Extract metadata
#         metadata = {
#             "source": os.path.basename(file_path),
#             "title": pdf_metadata.get("title", "Unknown Title"),
#             "author": pdf_metadata.get("author", "Unknown Author"),
#             "section_count": len(raw_text.split('\n')) if not pdf_metadata else len(doc),
#             "keyword_freq": extract_metadata(cleaned_text, file_path).get("keyword_freq", {}),
#             "position_weight": 0,
#             "section_weight": 0
#         }
        
#         # Save metadata
#         os.makedirs(metadata_dir, exist_ok=True)
#         metadata_output_path = os.path.join(metadata_dir, f"{base_name}.json")
#         with open(metadata_output_path, 'w', encoding='utf-8') as f:
#             json.dump(metadata, f, indent=4)
#         logging.debug(f"Saved metadata to {metadata_output_path}")

#         # Split text into chunks
#         chunks = []
#         for i in range(0, len(cleaned_text), chunk_size - chunk_overlap):
#             chunk = cleaned_text[i:i + chunk_size]
#             chunks.append(chunk)
        
#         # Save chunks
#         os.makedirs(chunks_dir, exist_ok=True)
#         chunk_output_path = os.path.join(chunks_dir, f"{base_name}_chunks.json")
#         with open(chunk_output_path, 'w', encoding='utf-8') as f:
#             json.dump({"chunks": chunks}, f)
#         logging.debug(f"Saved chunks to {chunk_output_path}")

#         # Generate embeddings
#         config = load_config('config.yaml')
#         model = SentenceTransformer(config['embedding']['model'], local_files_only=True)
#         embeddings = model.encode(chunks, show_progress_bar=True)
#         embeddings = np.array(embeddings).astype('float32')

#         if file_path.lower().endswith('.pdf'):
#             doc.close()

#         return text_output_path, metadata_output_path, chunk_output_path, chunks, embeddings
#     except Exception as e:
#         logging.error(f"Error processing {file_path}: {str(e)}")
#         return None, None, None, [], None

# def preprocess_directory(input_dir, text_dir, metadata_dir, chunks_dir=None):
#     """
#     Process all PDFs and text files in a directory, append embeddings to FAISS index and chunk mapping.
#     Args:
#         input_dir (str): Directory containing files.
#         text_dir (str): Directory to save cleaned texts.
#         metadata_dir (str): Directory to save metadata.
#         chunks_dir (str, optional): Directory to save chunks. Defaults to text_dir.
#     Returns:
#         list: List of processed file paths.
#     """
#     logging.debug(f"Preprocessing input_dir: {input_dir}, text_dir: {text_dir}, metadata_dir: {metadata_dir}, chunks_dir: {chunks_dir}")
#     config = load_config('config.yaml')
#     index_path = config['faiss']['index_path']
#     chunk_mapping_path = os.path.join(os.path.dirname(index_path), "chunk_mapping.json")
#     chunk_size = config['chunk']['size']
#     chunk_overlap = config['chunk']['overlap']
#     chunks_dir = chunks_dir or text_dir

#     # Load existing FAISS index and chunk mapping
#     index = None
#     existing_chunks = []
#     chunk_mapping = {"chunks": []}
#     if os.path.exists(index_path):
#         try:
#             index = faiss.read_index(index_path)
#             logging.debug(f"Loaded FAISS index from {index_path}")
#         except Exception as e:
#             logging.error(f"Failed to load FAISS index: {e}")
#             raise
#     if os.path.exists(chunk_mapping_path):
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
#                     chunk_mapping = {"chunks": converted_chunks}
#                 else:
#                     chunk_mapping = loaded_data
#                 existing_chunks = chunk_mapping.get("chunks", [])
#             logging.debug(f"Loaded {len(existing_chunks)} chunks from {chunk_mapping_path}")
#         except Exception as e:
#             logging.error(f"Failed to load chunk mapping: {e}")
#             raise

#     # Process files
#     processed_files = []
#     new_chunks = []
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith(('.pdf', '.txt')):
#             file_path = os.path.join(input_dir, filename)
#             if str(file_path) in {chunk["file"] for chunk in existing_chunks if isinstance(chunk, dict)}:
#                 logging.debug(f"Skipping already processed file: {file_path}")
#                 continue

#             text_path, meta_path, chunk_path, chunks, embeddings = preprocess_file(
#                 file_path, text_dir, metadata_dir, chunks_dir, chunk_size, chunk_overlap
#             )
#             if text_path and meta_path and chunk_path:
#                 processed_files.append(file_path)
#                 new_chunks.extend(chunks)
#                 for i, chunk in enumerate(chunks):
#                     chunk_mapping["chunks"].append({
#                         "file": file_path,
#                         "start": i * (chunk_size - chunk_overlap),
#                         "end": i * (chunk_size - chunk_overlap) + len(chunk),
#                         "text": chunk
#                     })
#                 if embeddings is not None:
#                     if index is None:
#                         dimension = embeddings.shape[1]
#                         index = faiss.IndexFlatL2(dimension)
#                     index.add(embeddings)
#                     logging.debug(f"Added embeddings for {file_path}")

#     if not processed_files:
#         logging.info(f"No new files to process in {input_dir}")
#         return [str(f) for f in chain(Path(input_dir).glob("*.[pP][dD][fF]"), Path(input_dir).glob("*.[tT][xX][tT]"))]

#     # Save updated FAISS index
#     try:
#         os.makedirs(os.path.dirname(index_path), exist_ok=True)
#         faiss.write_index(index, index_path)
#         logging.debug(f"Updated FAISS index at {index_path}")
#     except Exception as e:
#         logging.error(f"Failed to update FAISS index: {e}")
#         raise

#     # Save updated chunk mapping
#     try:
#         os.makedirs(os.path.dirname(chunk_mapping_path), exist_ok=True)
#         with open(chunk_mapping_path, "w") as f:
#             json.dump(chunk_mapping, f)
#         logging.debug(f"Updated chunk mapping at {chunk_mapping_path}")
#     except Exception as e:
#         logging.error(f"Failed to save chunk mapping: {e}")
#         raise

#     logging.info(f"Processed {len(processed_files)} files")
#     return processed_files



# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
# import logging
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# from sentence_transformers import SentenceTransformer
# import fitz  # PyMuPDF
# import faiss
# import numpy as np
# import json
# from pathlib import Path
# from itertools import chain
# from utils.config_loader import load_config
# from utils.text_utils import clean_text, get_keyword_frequency
# from utils.metadata_utils import extract_metadata
# from app.chunker import Chunker

# def preprocess_file(file_path, text_dir, metadata_dir, chunks_dir, chunk_size, chunk_overlap):
#     """
#     Extract text, metadata, and chunks from a PDF or text file, generate embeddings, and save to output directories.
#     """
#     try:
#         # Extract text
#         raw_text = ""
#         pdf_metadata = {}
#         if file_path.lower().endswith('.pdf'):
#             doc = fitz.open(file_path)
#             for page in doc:
#                 raw_text += page.get_text("text") + "\n"
#             pdf_metadata = doc.metadata
#         else:  # .txt
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 raw_text = f.read()

#         cleaned_text = clean_text(raw_text)
        
#         # Save cleaned text
#         os.makedirs(text_dir, exist_ok=True)
#         base_name = os.path.splitext(os.path.basename(file_path))[0]
#         text_output_path = os.path.join(text_dir, f"{base_name}.txt")
#         with open(text_output_path, 'w', encoding='utf-8') as f:
#             f.write(cleaned_text)
#         logging.debug(f"Saved cleaned text to {text_output_path}")

#         # Extract metadata
#         metadata = {
#             "source": os.path.basename(file_path),
#             "title": pdf_metadata.get("title", "Unknown Title"),
#             "author": pdf_metadata.get("author", "Unknown Author"),
#             "section_count": len(raw_text.split('\n')) if not pdf_metadata else len(doc),
#             "keyword_freq": get_keyword_frequency(cleaned_text)
#         }
        
#         # Save metadata
#         os.makedirs(metadata_dir, exist_ok=True)
#         metadata_output_path = os.path.join(metadata_dir, f"{base_name}.json")
#         with open(metadata_output_path, 'w', encoding='utf-8') as f:
#             json.dump(metadata, f, indent=4)
#         logging.debug(f"Saved metadata to {metadata_output_path}")

#         # Split text into chunks
#         chunker = Chunker(chunk_size, chunk_overlap)
#         chunks = chunker.chunk_text(cleaned_text, file_path)
#         chunk_texts = [chunk["text"] for chunk in chunks]
        
#         # Save chunks
#         os.makedirs(chunks_dir, exist_ok=True)
#         chunk_output_path = os.path.join(chunks_dir, f"{base_name}_chunks.json")
#         with open(chunk_output_path, 'w', encoding='utf-8') as f:
#             json.dump({"chunks": chunks}, f)
#         logging.debug(f"Saved chunks to {chunk_output_path}")

#         # Generate embeddings
#         config = load_config('config.yaml')
#         model = SentenceTransformer(config['embedding']['model'], local_files_only=True)
#         embeddings = model.encode(chunk_texts, show_progress_bar=True)
#         embeddings = np.array(embeddings).astype('float32')

#         if file_path.lower().endswith('.pdf'):
#             doc.close()

#         return text_output_path, metadata_output_path, chunk_output_path, chunks, embeddings
#     except Exception as e:
#         logging.error(f"Error processing {file_path}: {str(e)}")
#         return None, None, None, [], None

# def preprocess_directory(input_dir, text_dir, metadata_dir, chunks_dir=None):
#     """
#     Process all PDFs and text files in a directory, append embeddings to FAISS index and chunk mapping.
#     """
#     logging.debug(f"Preprocessing input_dir: {input_dir}, text_dir: {text_dir}, metadata_dir: {metadata_dir}, chunks_dir: {chunks_dir}")
#     config = load_config('config.yaml')
#     index_path = config['faiss']['index_path']
#     chunk_mapping_path = os.path.join(os.path.dirname(index_path), "chunk_mapping.json")
#     chunk_size = config['chunk']['size']
#     chunk_overlap = config['chunk']['overlap']
#     chunks_dir = chunks_dir or text_dir

#     # Load existing FAISS index and chunk mapping
#     index = None
#     existing_chunks = []
#     chunk_mapping = {"chunks": []}
#     if os.path.exists(index_path):
#         try:
#             index = faiss.read_index(index_path)
#             logging.debug(f"Loaded FAISS index from {index_path}")
#         except Exception as e:
#             logging.error(f"Failed to load FAISS index: {e}")
#             raise
#     if os.path.exists(chunk_mapping_path):
#         try:
#             with open(chunk_mapping_path, "r") as f:
#                 chunk_mapping = json.load(f)
#                 existing_chunks = chunk_mapping.get("chunks", [])
#             logging.debug(f"Loaded {len(existing_chunks)} chunks from {chunk_mapping_path}")
#         except Exception as e:
#             logging.error(f"Failed to load chunk mapping: {e}")
#             raise

#     # Process files
#     processed_files = []
#     new_chunks = []
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith(('.pdf', '.txt')):
#             file_path = os.path.join(input_dir, filename)
#             if str(file_path) in {chunk["file"] for chunk in existing_chunks if isinstance(chunk, dict)}:
#                 logging.debug(f"Skipping already processed file: {file_path}")
#                 continue

#             text_path, meta_path, chunk_path, chunks, embeddings = preprocess_file(
#                 file_path, text_dir, metadata_dir, chunks_dir, chunk_size, chunk_overlap
#             )
#             if text_path and meta_path and chunk_path:
#                 processed_files.append(file_path)
#                 new_chunks.extend(chunks)
#                 chunk_mapping["chunks"].extend(chunks)
#                 if embeddings is not None:
#                     if index is None:
#                         dimension = embeddings.shape[1]
#                         index = faiss.IndexFlatL2(dimension)
#                     index.add(embeddings)
#                     logging.debug(f"Added embeddings for {file_path}")

#     if not processed_files:
#         logging.info(f"No new files to process in {input_dir}")
#         return [str(f) for f in chain(Path(input_dir).glob("*.[pP][dD][fF]"), Path(input_dir).glob("*.[tT][xX][tT]"))]

#     # Save updated FAISS index
#     try:
#         os.makedirs(os.path.dirname(index_path), exist_ok=True)
#         faiss.write_index(index, index_path)
#         logging.debug(f"Updated FAISS index at {index_path}")
#     except Exception as e:
#         logging.error(f"Failed to update FAISS index: {e}")
#         raise

#     # Save updated chunk mapping
#     try:
#         os.makedirs(os.path.dirname(chunk_mapping_path), exist_ok=True)
#         with open(chunk_mapping_path, "w") as f:
#             json.dump(chunk_mapping, f)
#         logging.debug(f"Updated chunk mapping at {chunk_mapping_path}")
#     except Exception as e:
#         logging.error(f"Failed to save chunk mapping: {e}")
#         raise

#     logging.info(f"Processed {len(processed_files)} files")
#     return processed_files





# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"

# from pathlib import Path
# import pdfminer.high_level
# from utils.text_utils import clean_text, extract_keywords
# import json

# def extract_text_from_pdf(pdf_path):
#     try:
#         with open(pdf_path, 'rb') as file:
#             text = pdfminer.high_level.extract_text(file)
#         return text
#     except Exception as e:
#         return ""

# def chunk_text(text, chunk_size=1000, overlap=200):
#     chunks = []
#     start = 0
#     text = clean_text(text)
#     while start < len(text):
#         end = min(start + chunk_size, len(text))
#         chunks.append(text[start:end])
#         start = end - overlap
#     return chunks

# def extract_metadata(text, file_path):
#     keywords = extract_keywords(text, top_n=10)
#     return {
#         "keywords": keywords,
#         "title": os.path.basename(file_path).replace('.pdf', ''),
#         "section_weight": 1.0
#     }

# def preprocess_directory(input_dir, text_dir, metadata_dir, chunks_dir, force_reprocess=False):
#     input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pdf')]
#     processed_files = []
    
#     os.makedirs(text_dir, exist_ok=True)
#     os.makedirs(metadata_dir, exist_ok=True)
#     os.makedirs(chunks_dir, exist_ok=True)
    
#     for file_path in input_files:
#         output_text_path = os.path.join(text_dir, os.path.basename(file_path).replace('.pdf', '.txt'))
#         if os.path.exists(output_text_path) and not force_reprocess:
#             processed_files.append(file_path)
#             continue
        
#         text = extract_text_from_pdf(file_path)
#         if not text:
#             continue
        
#         cleaned_text = clean_text(text)
#         with open(output_text_path, 'w', encoding='utf-8') as f:
#             f.write(cleaned_text)
        
#         metadata = extract_metadata(cleaned_text, file_path)
#         metadata_path = os.path.join(metadata_dir, os.path.basename(file_path).replace('.pdf', '.json'))
#         with open(metadata_path, 'w', encoding='utf-8') as f:
#             json.dump(metadata, f, indent=2)
        
#         chunks = chunk_text(cleaned_text)
#         chunks_path = os.path.join(chunks_dir, os.path.basename(file_path).replace('.pdf', '_chunks.json'))
#         with open(chunks_path, 'w', encoding='utf-8') as f:
#             json.dump({"chunks": chunks}, f, indent=2)
        
#         processed_files.append(file_path)
    
#     return processed_files

# def main():
#     input_dir = "data/raw_pdfs"
#     text_dir = "data/cleaned_texts"
#     metadata_dir = "data/metadata"
#     chunks_dir = "data/chunks"
#     processed_files = preprocess_directory(input_dir, text_dir, metadata_dir, chunks_dir, force_reprocess=True)

# if __name__ == "__main__":
#     main()


# # -----------------------
# # working


# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
# import logging
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# from sentence_transformers import SentenceTransformer
# import fitz  # PyMuPDF
# import faiss
# import numpy as np
# import json
# from pathlib import Path
# from itertools import chain
# from utils.config_loader import load_config
# from utils.text_utils import clean_text
# from utils.metadata_utils import extract_metadata

# def preprocess_file(file_path, text_dir, metadata_dir, chunks_dir, chunk_size, chunk_overlap):
#     """
#     Extract text, metadata, and chunks from a PDF or text file, generate embeddings, and save to output directories.
#     Args:
#         file_path (str): Path to the file.
#         text_dir (str): Directory to save cleaned text.
#         metadata_dir (str): Directory to save metadata.
#         chunks_dir (str): Directory to save chunks.
#         chunk_size (int): Size of each text chunk.
#         chunk_overlap (int): Overlap between chunks.
#     Returns:
#         tuple: (text_output_path, metadata_output_path, chunk_output_path, chunks, embeddings)
#     """
#     try:
#         # Extract text
#         raw_text = ""
#         pdf_metadata = {}
#         if file_path.lower().endswith('.pdf'):
#             doc = fitz.open(file_path)
#             for page in doc:
#                 raw_text += page.get_text("text") + "\n"
#             pdf_metadata = doc.metadata
#         else:  # .txt
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 raw_text = f.read()

#         cleaned_text = clean_text(raw_text)
        
#         # Save cleaned text
#         os.makedirs(text_dir, exist_ok=True)
#         base_name = os.path.splitext(os.path.basename(file_path))[0]
#         text_output_path = os.path.join(text_dir, f"{base_name}.txt")
#         with open(text_output_path, 'w', encoding='utf-8') as f:
#             f.write(cleaned_text)
#         logging.debug(f"Saved cleaned text to {text_output_path}")

#         # Extract metadata
#         metadata = {
#             "source": os.path.basename(file_path),
#             "title": pdf_metadata.get("title", "Unknown Title"),
#             "author": pdf_metadata.get("author", "Unknown Author"),
#             "section_count": len(raw_text.split('\n')) if not pdf_metadata else len(doc),
#             "keyword_freq": extract_metadata(cleaned_text, file_path).get("keyword_freq", {}),
#             "position_weight": 0,
#             "section_weight": 0
#         }
        
#         # Save metadata
#         os.makedirs(metadata_dir, exist_ok=True)
#         metadata_output_path = os.path.join(metadata_dir, f"{base_name}.json")
#         with open(metadata_output_path, 'w', encoding='utf-8') as f:
#             json.dump(metadata, f, indent=4)
#         logging.debug(f"Saved metadata to {metadata_output_path}")

#         # Split text into chunks
#         chunks = []
#         for i in range(0, len(cleaned_text), chunk_size - chunk_overlap):
#             chunk = cleaned_text[i:i + chunk_size]
#             chunks.append(chunk)
        
#         # Save chunks
#         os.makedirs(chunks_dir, exist_ok=True)
#         chunk_output_path = os.path.join(chunks_dir, f"{base_name}_chunks.json")
#         with open(chunk_output_path, 'w', encoding='utf-8') as f:
#             json.dump({"chunks": chunks}, f)
#         logging.debug(f"Saved chunks to {chunk_output_path}")

#         # Generate embeddings
#         config = load_config('config.yaml')
#         model = SentenceTransformer(config['embedding']['model'], local_files_only=True)
#         embeddings = model.encode(chunks, show_progress_bar=True)
#         embeddings = np.array(embeddings).astype('float32')

#         if file_path.lower().endswith('.pdf'):
#             doc.close()

#         return text_output_path, metadata_output_path, chunk_output_path, chunks, embeddings
#     except Exception as e:
#         logging.error(f"Error processing {file_path}: {str(e)}")
#         return None, None, None, [], None

# def preprocess_directory(input_dir, text_dir, metadata_dir, chunks_dir=None):
#     """
#     Process all PDFs and text files in a directory, append embeddings to FAISS index and chunk mapping.
#     Args:
#         input_dir (str): Directory containing files.
#         text_dir (str): Directory to save cleaned texts.
#         metadata_dir (str): Directory to save metadata.
#         chunks_dir (str, optional): Directory to save chunks. Defaults to text_dir.
#     Returns:
#         list: List of processed file paths.
#     """
#     logging.debug(f"Preprocessing input_dir: {input_dir}, text_dir: {text_dir}, metadata_dir: {metadata_dir}, chunks_dir: {chunks_dir}")
#     config = load_config('config.yaml')
#     index_path = config['faiss']['index_path']
#     chunk_mapping_path = os.path.join(os.path.dirname(index_path), "chunk_mapping.json")
#     chunk_size = config['chunk']['size']
#     chunk_overlap = config['chunk']['overlap']
#     chunks_dir = chunks_dir or text_dir

#     # Load existing FAISS index and chunk mapping
#     index = None
#     existing_chunks = []
#     chunk_mapping = {"chunks": []}
#     if os.path.exists(index_path):
#         try:
#             index = faiss.read_index(index_path)
#             logging.debug(f"Loaded FAISS index from {index_path}")
#         except Exception as e:
#             logging.error(f"Failed to load FAISS index: {e}")
#             raise
#     if os.path.exists(chunk_mapping_path):
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
#                     chunk_mapping = {"chunks": converted_chunks}
#                 else:
#                     chunk_mapping = loaded_data
#                 existing_chunks = chunk_mapping.get("chunks", [])
#             logging.debug(f"Loaded {len(existing_chunks)} chunks from {chunk_mapping_path}")
#         except Exception as e:
#             logging.error(f"Failed to load chunk mapping: {e}")
#             raise

#     # Process files
#     processed_files = []
#     new_chunks = []
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith(('.pdf', '.txt')):
#             file_path = os.path.join(input_dir, filename)
#             if str(file_path) in {chunk["file"] for chunk in existing_chunks if isinstance(chunk, dict)}:
#                 logging.debug(f"Skipping already processed file: {file_path}")
#                 continue

#             text_path, meta_path, chunk_path, chunks, embeddings = preprocess_file(
#                 file_path, text_dir, metadata_dir, chunks_dir, chunk_size, chunk_overlap
#             )
#             if text_path and meta_path and chunk_path:
#                 processed_files.append(file_path)
#                 new_chunks.extend(chunks)
#                 for i, chunk in enumerate(chunks):
#                     chunk_mapping["chunks"].append({
#                         "file": file_path,
#                         "start": i * (chunk_size - chunk_overlap),
#                         "end": i * (chunk_size - chunk_overlap) + len(chunk),
#                         "text": chunk
#                     })
#                 if embeddings is not None:
#                     if index is None:
#                         dimension = embeddings.shape[1]
#                         index = faiss.IndexFlatL2(dimension)
#                     index.add(embeddings)
#                     logging.debug(f"Added embeddings for {file_path}")

#     if not processed_files:
#         logging.info(f"No new files to process in {input_dir}")
#         return [str(f) for f in chain(Path(input_dir).glob("*.[pP][dD][fF]"), Path(input_dir).glob("*.[tT][xX][tT]"))]

#     # Save updated FAISS index
#     try:
#         os.makedirs(os.path.dirname(index_path), exist_ok=True)
#         faiss.write_index(index, index_path)
#         logging.debug(f"Updated FAISS index at {index_path}")
#     except Exception as e:
#         logging.error(f"Failed to update FAISS index: {e}")
#         raise

#     # Save updated chunk mapping
#     try:
#         os.makedirs(os.path.dirname(chunk_mapping_path), exist_ok=True)
#         with open(chunk_mapping_path, "w") as f:
#             json.dump(chunk_mapping, f)
#         logging.debug(f"Updated chunk mapping at {chunk_mapping_path}")
#     except Exception as e:
#         logging.error(f"Failed to save chunk mapping: {e}")
#         raise

#     logging.info(f"Processed {len(processed_files)} files")
#     return processed_files


# ----------------------------

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
import faiss
import numpy as np
import json
from pathlib import Path
from itertools import chain
from utils.config_loader import load_config
from utils.text_utils import clean_text
from utils.metadata_utils import extract_metadata
import nltk
nltk.data.path.append("/Users/nimishgupta/Documents/rag_doc/nltk_data")

def preprocess_file(file_path, text_dir, metadata_dir, chunks_dir, chunk_size, chunk_overlap):
    """
    Extract text, metadata, and chunks from a PDF or text file, generate embeddings, and save to output directories.
    Args:
        file_path (str): Path to the file.
        text_dir (str): Directory to save cleaned text.
        metadata_dir (str): Directory to save metadata.
        chunks_dir (str): Directory to save chunks.
        chunk_size (int): Target size of each text chunk.
        chunk_overlap (int): Overlap between chunks.
    Returns:
        tuple: (text_output_path, metadata_output_path, chunk_output_path, chunks, embeddings, chunk_mapping_entries)
    """
    try:
        # Extract text
        raw_text = ""
        pdf_metadata = {}
        pdf_page_count = 0
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            for page in doc:
                text = page.get_text("text")
                raw_text += text + "\n"
            pdf_metadata = doc.metadata
            pdf_page_count = len(doc)
            if len(raw_text.strip()) < 100:  # Fallback to pdfminer.six
                logging.debug(f"PyMuPDF extracted minimal text ({len(raw_text)} chars) from {file_path}, trying pdfminer.six")
                raw_text = extract_text(file_path, maxpages=0)
            doc.close()
        else:  # .txt
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

        cleaned_text = clean_text(raw_text)
        if not cleaned_text.strip():
            logging.error(f"No usable text extracted from {file_path}")
            return None, None, None, [], None, []

        # Save cleaned text
        os.makedirs(text_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        text_output_path = os.path.join(text_dir, f"{base_name}.txt")
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        logging.debug(f"Saved cleaned text to {text_output_path}")

        # Extract document-level metadata
        doc_metadata = {
            "source": os.path.basename(file_path),
            "title": pdf_metadata.get("title", "Unknown Title"),
            "author": pdf_metadata.get("author", "Unknown Author"),
            "section_count": len(raw_text.split('\n')) if not pdf_metadata else pdf_page_count,
            "keyword_freq": extract_metadata(cleaned_text, file_path).get("keyword_freq", {}),
            "position_weight": 0,
            "section_weight": 0
        }

        # Save document metadata
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_output_path = os.path.join(metadata_dir, f"{base_name}.json")
        with open(metadata_output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_metadata, f, indent=4)
        logging.debug(f"Saved metadata to {metadata_output_path}")

        # Semantic chunking
        sentences = nltk.sent_tokenize(cleaned_text)
        chunks = []
        chunk_metadata = []
        current_chunk = []
        current_length = 0
        chunk_start = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    chunk_metadata.append({
                        "section_weight": 1.0 / (len(chunks) + 1),
                        "keyword_freq": extract_metadata(chunk_text, file_path).get("keyword_freq", {})
                    })
                    overlap_text = chunk_text[-chunk_overlap:] if len(chunk_text) > chunk_overlap else ""
                    current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                    current_length = len(overlap_text) + sentence_length
                    chunk_start = chunk_start + len(chunk_text) - len(overlap_text)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length

        # Handle the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            chunk_metadata.append({
                "section_weight": 1.0 / (len(chunks) + 1),
                "keyword_freq": extract_metadata(chunk_text, file_path).get("keyword_freq", {})
            })

        logging.debug(f"Created {len(chunks)} chunks from text of length {len(cleaned_text)}")

        # Save chunks
        os.makedirs(chunks_dir, exist_ok=True)
        chunk_output_path = os.path.join(chunks_dir, f"{base_name}_chunks.json")
        with open(chunk_output_path, 'w', encoding='utf-8') as f:
            json.dump({"chunks": chunks}, f)
        logging.debug(f"Saved chunks to {chunk_output_path}")

        # Generate embeddings
        config = load_config('config.yaml')
        model = SentenceTransformer(config['embedding']['model'], local_files_only=True)
        embeddings = model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Create chunk mapping entries
        chunk_mapping_entries = []
        current_pos = 0
        for i, chunk in enumerate(chunks):
            chunk_mapping_entries.append({
                "file": file_path,
                "start": current_pos,
                "end": current_pos + len(chunk),
                "text": chunk,
                "metadata": chunk_metadata[i]
            })
            current_pos += len(chunk) - (chunk_overlap if i < len(chunks) - 1 else 0)

        return text_output_path, metadata_output_path, chunk_output_path, chunks, embeddings, chunk_mapping_entries
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None, None, None, [], None, []

def preprocess_directory(input_dir, text_dir, metadata_dir, chunks_dir=None):
    """
    Process all PDFs and text files in a directory, append embeddings to FAISS index and chunk mapping.
    Args:
        input_dir (str): Directory containing files.
        text_dir (str): Directory to save cleaned texts.
        metadata_dir (str): Directory to save metadata.
        chunks_dir (str, optional): Directory to save chunks. Defaults to text_dir.
    Returns:
        list: List of processed file paths.
    """
    logging.debug(f"Preprocessing input_dir: {input_dir}, text_dir: {text_dir}, metadata_dir: {metadata_dir}, chunks_dir: {chunks_dir}")
    config = load_config('config.yaml')
    index_path = config['faiss']['index_path']
    chunk_mapping_path = os.path.join(os.path.dirname(index_path), "chunk_mapping.json")
    chunk_size = config['chunk']['size']
    chunk_overlap = config['chunk']['overlap']
    chunks_dir = chunks_dir or text_dir

    # Load existing FAISS index and chunk mapping
    index = None
    existing_chunks = []
    chunk_mapping = {"chunks": []}
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            logging.debug(f"Loaded FAISS index from {index_path}")
        except Exception as e:
            logging.error(f"Failed to load FAISS index: {e}")
            raise
    if os.path.exists(chunk_mapping_path):
        try:
            with open(chunk_mapping_path, "r") as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                    logging.warning(f"Converting list-based chunk_mapping.json to dictionary format")
                    converted_chunks = []
                    for chunk in loaded_data:
                        if isinstance(chunk, list) and len(chunk) >= 4:
                            converted_chunks.append({
                                "file": chunk[0],
                                "start": chunk[1],
                                "end": chunk[2],
                                "text": chunk[3],
                                "metadata": {}
                            })
                        else:
                            logging.warning(f"Skipping invalid chunk entry: {chunk}")
                    chunk_mapping = {"chunks": converted_chunks}
                else:
                    chunk_mapping = loaded_data
                existing_chunks = chunk_mapping.get("chunks", [])
            logging.debug(f"Loaded {len(existing_chunks)} chunks from {chunk_mapping_path}")
        except Exception as e:
            logging.error(f"Failed to load chunk mapping: {e}")
            raise

    # Process files
    processed_files = []
    new_chunks = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.pdf', '.txt')):
            file_path = os.path.join(input_dir, filename)
            if str(file_path) in {chunk["file"] for chunk in existing_chunks if isinstance(chunk, dict)}:
                logging.debug(f"Skipping already processed file: {file_path}")
                continue

            text_path, meta_path, chunk_path, chunks, embeddings, chunk_mapping_entries = preprocess_file(
                file_path, text_dir, metadata_dir, chunks_dir, chunk_size, chunk_overlap
            )
            if text_path and meta_path and chunk_path:
                processed_files.append(file_path)
                new_chunks.extend(chunks)
                chunk_mapping["chunks"].extend(chunk_mapping_entries)
                if embeddings is not None:
                    if index is None:
                        dimension = embeddings.shape[1]
                        index = faiss.IndexFlatL2(dimension)
                    index.add(embeddings)
                    logging.debug(f"Added embeddings for {file_path}")

    if not processed_files:
        logging.info(f"No new files to process in {input_dir}")
        return [str(f) for f in chain(Path(input_dir).glob("*.[pP][dD][fF]"), Path(input_dir).glob("*.[tT][xX][tT]"))]

    # Save updated FAISS index
    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        logging.debug(f"Updated FAISS index at {index_path}")
    except Exception as e:
        logging.error(f"Failed to update FAISS index: {e}")
        raise

    # Save updated chunk mapping
    try:
        os.makedirs(os.path.dirname(chunk_mapping_path), exist_ok=True)
        with open(chunk_mapping_path, "w") as f:
            json.dump(chunk_mapping, f)
        logging.debug(f"Updated chunk mapping at {chunk_mapping_path}")
    except Exception as e:
        logging.error(f"Failed to save chunk mapping: {e}")
        raise

    logging.info(f"Processed {len(processed_files)} files")
    return processed_files