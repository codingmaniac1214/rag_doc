# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
# import logging
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

# # Set custom nltk data path
# nltk.data.path.append("/Users/nimishgupta/Documents/rag_doc/nltk_data")

# def clean_text(text):
#     """
#     Clean text by removing unwanted characters and normalizing.
#     Args:
#         text (str): Raw text input.
#     Returns:
#         str: Cleaned text.
#     """
#     try:
#         text = re.sub(r'\s+', ' ', text)
#         text = re.sub(r'[^\w\s.]', '', text)
#         text = text.strip()
#         logging.debug(f"Cleaned text to {len(text)} characters")
#         return text
#     except Exception as e:
#         logging.error(f"Error cleaning text: {e}")
#         return ""

# def get_keyword_frequency(text):
#     """
#     Calculate frequency of keywords in text.
#     Args:
#         text (str): Input text.
#     Returns:
#         dict: Dictionary of keyword frequencies.
#     """
#     try:
#         stop_words = set(stopwords.words('english'))
#         words = word_tokenize(text.lower())
#         keywords = [word for word in words if word.isalnum() and word not in stop_words]
#         freq = {}
#         for word in keywords:
#             freq[word] = freq.get(word, 0) + 1
#         logging.debug(f"Extracted {len(freq)} keyword frequencies")
#         return freq
#     except Exception as e:
#         logging.error(f"Error extracting keyword frequency: {e}")
#         return {}

# def extract_keywords(text, top_n=10):
#     """
#     Extract top N keywords from text based on frequency.
#     Args:
#         text (str): Input text.
#         top_n (int): Number of top keywords to return.
#     Returns:
#         list: List of top keywords.
#     """
#     try:
#         freq = get_keyword_frequency(text)
#         top_keywords = [word for word, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]
#         logging.debug(f"Extracted {len(top_keywords)} keywords: {top_keywords}")
#         return top_keywords
#     except Exception as e:
#         logging.error(f"Error extracting keywords: {e}")
#         return []





import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.data.path.append("/Users/nimishgupta/Documents/rag_doc/nltk_data")

def clean_text(text):
    """
    Clean text by removing unwanted characters and normalizing.
    Args:
        text (str): Input text.
    Returns:
        str: Cleaned text.
    """
    try:
        # Preserve numbers, technical terms, and special characters
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        text = text.strip()
        return text
    except Exception as e:
        logging.error(f"Error cleaning text: {str(e)}")
        return text

def get_keyword_frequency(text):
    """
    Extract keyword frequency from text.
    Args:
        text (str): Input text.
    Returns:
        dict: Dictionary of keyword frequencies.
    """
    try:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        keywords = [word for word in words if word.isalnum() and word not in stop_words]
        return {word: keywords.count(word) for word in set(keywords)}
    except Exception as e:
        logging.error(f"Error extracting keyword frequency: {str(e)}")
        return {}

def extract_keywords(text, top_n=10):
    """
    Extract top keywords from text.
    Args:
        text (str): Input text.
        top_n (int): Number of keywords to return.
    Returns:
        list: List of top keywords.
    """
    try:
        keyword_freq = get_keyword_frequency(text)
        return sorted(keyword_freq, key=keyword_freq.get, reverse=True)[:top_n]
    except Exception as e:
        logging.error(f"Error extracting keywords: {str(e)}")
        return []