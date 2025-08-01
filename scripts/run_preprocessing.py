# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# from pathlib import Path
# import os

# # Set HF_HOME using a relative path to the models/ folder
# os.environ["HF_HOME"] = str(Path("models").resolve())
# import logging
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# from app.preprocess import preprocess_directory
# from utils.config_loader import load_config

# def main():
#     config = load_config('config.yaml')
#     input_dir = config.get('source_dir', 'data/raw_pdfs')
#     text_dir = config.get('cleaned_texts_dir', 'data/cleaned_texts')
#     metadata_dir = config.get('metadata_dir', 'data/metadata')
#     chunks_dir = config.get('chunks_dir', 'data/chunks')

#     logging.info(f"Starting preprocessing: input_dir={input_dir}, text_dir={text_dir}, metadata_dir={metadata_dir}, chunks_dir={chunks_dir}")
#     processed_files = preprocess_directory(input_dir, text_dir, metadata_dir, chunks_dir)
#     logging.info(f"Processed {len(processed_files)} files: {processed_files}")

# if __name__ == "__main__":
#     main()



import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(Path("models").resolve())
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", filename="log.log")
from app.preprocess import preprocess_directory
from utils.config_loader import load_config


def main(product="default"):
   """
   Run preprocessing for all files in the input directory for the specified product.
   Args:
       product (str): The product name to process (e.g., 'default', 'abc').
   """
   config = load_config('config.yaml')
   input_dir = os.path.join(config['base_data_dir'], product, "raw_pdfs")
   text_dir = os.path.join(config['base_data_dir'], product, "cleaned_texts")
   metadata_dir = os.path.join(config['base_data_dir'], product, "metadata")
   chunks_dir = os.path.join(config['base_data_dir'], product, "chunks")


   logging.info(f"Starting preprocessing for product '{product}': input_dir={input_dir}, text_dir={text_dir}, metadata_dir={metadata_dir}, chunks_dir={chunks_dir}")
   try:
       processed_files = preprocess_directory(input_dir, text_dir, metadata_dir, chunks_dir)
       logging.info(f"Processed {len(processed_files)} files for product '{product}': {processed_files}")
   except Exception as e:
       logging.error(f"Preprocessing failed for product '{product}': {str(e)}")
       raise


if __name__ == "__main__":
   product = sys.argv[1] if len(sys.argv) > 1 else "default"
   main(product)



