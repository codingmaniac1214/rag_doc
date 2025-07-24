import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.embedder import Embedder
from utils.config_loader import load_config

def main():
    config = load_config('config.yaml')
    chunk_dir = 'data/chunks'
    embedder = Embedder(config['embedding']['model'], config['faiss']['index_path'])
    
    chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith('.json')]
    embedder.create_index(chunk_files)

if __name__ == "__main__":
    main()