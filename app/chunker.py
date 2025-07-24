import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
import spacy
from utils.text_utils import clean_text, get_keyword_frequency

class Chunker:
    def __init__(self, chunk_size=1000, chunk_overlap=500):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logging.debug("SpaCy model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load SpaCy model: {e}")
            raise

    def chunk_text(self, text, file_path):
        """
        Split text into semantic-aware chunks using spaCy.
        Args:
            text (str): Input text.
            file_path (str): Source file path for metadata.
        Returns:
            list: List of dictionaries with chunk text and metadata.
        """
        try:
            cleaned_text = clean_text(text)
            doc = self.nlp(cleaned_text)
            chunks = []
            current_chunk = ""
            current_start = 0
            base_name = os.path.basename(file_path)

            for sent in doc.sents:
                sent_text = sent.text.strip()
                if len(current_chunk) + len(sent_text) <= self.chunk_size:
                    current_chunk += sent_text + " "
                else:
                    if current_chunk:
                        chunk_text = current_chunk.strip()
                        chunks.append({
                            "file": file_path,
                            "start": current_start,
                            "end": current_start + len(chunk_text),
                            "text": chunk_text,
                            "section_weight": 1.0 if "introduction" in chunk_text.lower() else 0.5,
                            "keyword_freq": get_keyword_frequency(chunk_text)
                        })
                        current_start = current_start + len(chunk_text) - self.chunk_overlap
                        current_chunk = current_chunk[-self.chunk_overlap:] + sent_text + " "
                    else:
                        current_chunk = sent_text + " "

            if current_chunk:
                chunk_text = current_chunk.strip()
                chunks.append({
                    "file": file_path,
                    "start": current_start,
                    "end": current_start + len(chunk_text),
                    "text": chunk_text,
                    "section_weight": 1.0 if "introduction" in chunk_text.lower() else 0.5,
                    "keyword_freq": get_keyword_frequency(chunk_text)
                })

            logging.debug(f"Generated {len(chunks)} chunks for {file_path}")
            return chunks
        except Exception as e:
            logging.error(f"Error chunking {file_path}: {e}")
            return []