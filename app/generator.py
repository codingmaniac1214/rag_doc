import ollama
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_answer(query, chunks, model_name="llama3"):
    """
    Generate an answer using a local LLM and retrieved chunks.
    Args:
        query (str): The user query.
        chunks (list): List of chunk texts from Retriever.
        model_name (str): Name of the LLM model.
    Returns:
        str: Generated answer.
    """
    try:
        # Prepare context from chunks
        context = "\n\n".join(chunks) if chunks else "No relevant information found."
        prompt = (
            f"You are a helpful assistant. Answer the following question based solely on the provided context. "
            f"Do not use external knowledge or make assumptions beyond the context. If the context is insufficient, "
            f"say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        logging.debug(f"Prompt sent to LLM:\n{prompt}")
        
        # Call Ollama LLM with conservative parameters
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": 0.2,  # Reduce creativity
                "top_p": 0.9,
                "max_tokens": 500
            }
        )
        
        answer = response['response'].strip()
        logging.debug(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn't generate an answer due to an error."