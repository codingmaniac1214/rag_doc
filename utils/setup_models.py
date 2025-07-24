import os
import nltk
from sentence_transformers import SentenceTransformer
import spacy
import ollama
from utils.config_loader import load_config

def setup_nltk_models(download_dir="nltk_data"):
    """Download NLTK models to project-local directory and configure path."""
    download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    os.environ["NLTK_DATA"] = download_dir
    nltk.data.path.append(download_dir)
    for model in ["punkt", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{model}")
            print(f"NLTK model '{model}' found in {download_dir}")
        except LookupError:
            print(f"Downloading NLTK model '{model}' to {download_dir}")
            nltk.download(model, download_dir=download_dir)

def setup_sentence_transformers(model_name, download_dir="sentence_transformers"):
    """Download SentenceTransformers model to project-local directory."""
    download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    try:
        SentenceTransformer(model_name, cache_folder=download_dir)
        print(f"SentenceTransformers model '{model_name}' found in {download_dir}")
    except Exception:
        print(f"Downloading SentenceTransformers model '{model_name}' to {download_dir}")
        SentenceTransformer(model_name, cache_folder=download_dir)

def setup_spacy_model(model_name="en_core_web_sm", download_dir="spacy_data"):
    """Download spaCy model to project-local directory."""
    download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    model_path = os.path.join(download_dir, model_name)
    try:
        spacy.load(model_path)
        print(f"spaCy model '{model_name}' found in {model_path}")
    except OSError:
        print(f"Downloading spaCy model '{model_name}' to {download_dir}")
        os.system(f"python -m spacy download {model_name} --target {download_dir}")

def verify_ollama_model(model_name):
    """Verify or pull Ollama model."""
    try:
        ollama.show(model_name)
        print(f"Ollama model '{model_name}' is ready")
    except Exception:
        print(f"Pulling Ollama model '{model_name}'")
        ollama.pull(model_name)
        print(f"Ollama model '{model_name}' downloaded")

if __name__ == "__main__":
    config = load_config("config.yaml")
    embedding_model = config["embedding"]["model"]
    llm_model = config["llm"]["model"]
    spacy_model = config["spacy"]["model"]

    print("Setting up models in project-local directories...")
    setup_nltk_models()
    setup_sentence_transformers(embedding_model)
    setup_spacy_model(spacy_model)
    verify_ollama_model(llm_model)
    print("All models set up successfully. System is ready for offline use.")