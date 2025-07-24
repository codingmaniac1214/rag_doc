import re
import os
from utils.text_utils import get_keyword_frequency

def extract_metadata(text, pdf_path):
    """
    Extract metadata from text and PDF file.
    """
    try:
        metadata = {
            "source": os.path.basename(pdf_path),
            "title": extract_title(text),
            "author": extract_author(text),
            "section_count": len(re.findall(r'^(#+)\s', text, re.MULTILINE)),
            "keyword_freq": get_keyword_frequency(text),
            "position_weight": 0,
            "section_weight": 0
        }
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return {}

def extract_title(text):
    """
    Extract the title from text (heuristic-based).
    """
    lines = text.split('\n', 1)
    return lines[0].strip() if lines else "Unknown Title"

def extract_author(text):
    """
    Extract the author from text (heuristic-based).
    """
    author_match = re.search(r'(?:author|by)\s*:\s*([^\n]+)', text, re.I)
    return author_match.group(1).strip() if author_match else "Unknown Author"