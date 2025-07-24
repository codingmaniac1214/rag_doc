import argparse
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
from app.retriever import Retriever
from app.generator import generate_answer
from utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="Run query pipeline")
    parser.add_argument('--query', type=str, required=True, help="User query")
    args = parser.parse_args()
    
    config = load_config('config.yaml')
    try:
        retriever = Retriever(
            config['embedding']['model'],
            config['faiss']['index_path'],
            config['relevance_model']['path']
        )
    except Exception as e:
        logging.error(f"Failed to initialize retriever: {e}")
        return
    
    # Retrieve top chunks
    chunks = retriever.retrieve(args.query, top_k=5)
    logging.debug(f"Retrieved chunks: {chunks}")
    
    # Generate answer
    answer = generate_answer(args.query, chunks, model_name=config['llm']['model'])
    print(f"Query: {args.query}\nAnswer: {answer}")

if __name__ == "__main__":
    main()