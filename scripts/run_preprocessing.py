import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
from app.preprocess import preprocess_directory
from utils.config_loader import load_config

def main():
    config = load_config('config.yaml')
    input_dir = config.get('source_dir', 'data/raw_pdfs')
    text_dir = config.get('cleaned_texts_dir', 'data/cleaned_texts')
    metadata_dir = config.get('metadata_dir', 'data/metadata')
    chunks_dir = config.get('chunks_dir', 'data/chunks')

    logging.info(f"Starting preprocessing: input_dir={input_dir}, text_dir={text_dir}, metadata_dir={metadata_dir}, chunks_dir={chunks_dir}")
    processed_files = preprocess_directory(input_dir, text_dir, metadata_dir, chunks_dir)
    logging.info(f"Processed {len(processed_files)} files: {processed_files}")

if __name__ == "__main__":
    main()