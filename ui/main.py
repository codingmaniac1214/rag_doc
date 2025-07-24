# working


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
import streamlit as st
from app.retriever import Retriever
from app.generator import generate_answer
from utils.config_loader import load_config
from streamlit_chat import message

# Environment & Path Setup
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Streamlit Config
st.set_page_config(page_title="📄 Offline Document Q&A", layout="wide")

# Sidebar: Load config and file uploader
with st.sidebar:
    st.title("⚙️ Settings")
    config = load_config('config.yaml')

    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("🧾 Process Documents"):
        if uploaded_files:
            with st.spinner("Processing uploaded documents..."):
                for file in uploaded_files:
                    with open(os.path.join("docs", file.name), "wb") as f:
                        f.write(file.read())
                st.success("Files saved and ready for retrieval!")
        else:
            st.warning("Please upload at least one document.")

    st.markdown("---")
    st.markdown("### 🔍 Retrieval Settings")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=100, value=5)

    st.markdown("---")
    st.markdown("### 🎨 Theme")
    theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)
    if theme == "Dark":
        st.markdown("""
            <style>
            body { background-color: #111; color: #eee; }
            .stButton>button { background-color: #333; color: white; }
            </style>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.markdown(f"**Embedding Model:** `{config['embedding']['model']}`")
    st.markdown(f"**LLM:** `{config['llm']['model']}`")
    rel_model = config['relevance_model'].get('path', 'None')
    st.markdown(f"**Relevance Model:** `{rel_model}`")

# Main UI
st.title("📚 Ask Your Documents (Offline)")

retriever = Retriever(
    config['embedding']['model'],                # 2nd: embedding model
    config['faiss']['index_path'],               # 3rd: faiss index path
    config['relevance_model'].get('path')        # 4th: optional relevance model
)

# Maintain session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Query Input and Answer Generation (First Block)
query = st.text_input("Enter your question:")
if st.button("Ask"):
    if query:
        with st.spinner("Retrieving relevant chunks..."):
            chunks = retriever.retrieve(query, top_k=top_k)
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, chunks, model_name=config['llm']['model'])
        st.write("**Answer:**")
        st.write(answer)

        # Save to chat history
        st.session_state.chat_history.append((query, answer))
    else:
        st.write("Please enter a question.")

# Display chat history
st.subheader("💬 Chat History")
for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
    message(f"**You:** {q}", is_user=True, key=f"user_{i}")
    message(f"**AI:** {a}", is_user=False, key=f"ai_{i}")

# Clear history
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")


# import sys
# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
# import logging
# logging.basicConfig(level=logging.DEBUG)
# import streamlit as st
# from app.retriever import Retriever
# from app.generator import generate_answer
# from utils.config_loader import load_config
# from streamlit_chat import message
# from pathlib import Path
# from app.preprocess import preprocess_directory

# st.set_page_config(page_title="📄 Offline Document Q&A", layout="wide")

# # Sidebar: Load config and file uploader
# with st.sidebar:
#     st.title("⚙️ Settings")
#     config = load_config('config.yaml')

#     st.markdown("### 📂 Upload Documents")
#     source_dir = config.get('source_dir', 'raw_pdfs')  # Default to 'raw_pdfs'
#     st.markdown(f"**Uploading to:** `{source_dir}`")
#     uploaded_files = st.file_uploader("Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True)
#     if st.button("🧾 Preprocess Documents"):
#         if uploaded_files:
#             with st.spinner("Saving and preprocessing uploaded documents..."):
#                 # Save uploaded files to source directory
#                 os.makedirs(source_dir, exist_ok=True)
#                 for file in uploaded_files:
#                     file_path = os.path.join(source_dir, file.name)
#                     with open(file_path, "wb") as f:
#                         f.write(file.read())
#                 # Run preprocessing to append to existing embeddings and data
#                 try:
#                     processed_files = preprocess_directory(source_dir)
#                     st.success(f"Documents preprocessed successfully! Added {len(processed_files)} files to existing embeddings and data.")
#                     # Reinitialize retriever with updated index
#                     st.session_state.retriever = Retriever(
#                         config['embedding']['model'],
#                         config['faiss']['index_path'],
#                         config['relevance_model'].get('path')
#                     )
#                 except Exception as e:
#                     st.error(f"Preprocessing failed: {e}")
#                     logging.error(f"Preprocessing error: {e}")
#         else:
#             st.warning("Please upload at least one document.")

#     st.markdown("---")
#     st.markdown("### 🔍 Retrieval Settings")
#     top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5)

#     st.markdown("---")
#     st.markdown("### 🎨 Theme")
#     theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)
#     if theme == "Dark":
#         st.markdown("""
#             <style>
#             body { background-color: #111; color: #eee; }
#             .stButton>button { background-color: #333; color: white; }
#             </style>
#         """, unsafe_allow_html=True)

#     st.markdown("---")
#     st.markdown("### 🤖 Model Info")
#     st.markdown(f"**Embedding Model:** `{config['embedding']['model']}`")
#     st.markdown(f"**LLM:** `{config['llm']['model']}`")
#     rel_model = config['relevance_model'].get('path', 'None')
#     st.markdown(f"**Relevance Model:** `{rel_model}`")

# # Main UI
# st.title("📚 Ask Your Documents (Offline)")

# # Initialize retriever (default, updated after preprocessing)
# if "retriever" not in st.session_state:
#     try:
#         st.session_state.retriever = Retriever(
#             config['embedding']['model'],
#             config['faiss']['index_path'],
#             config['relevance_model'].get('path')
#         )
#     except Exception as e:
#         st.error(f"Failed to initialize retriever: {e}")
#         logging.error(f"Retriever initialization error: {e}")

# # Maintain session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# query = st.text_input("💬 Ask a question from your documents:")

# if st.button("🔎 Ask"):
#     if query.strip():
#         if "retriever" in st.session_state:
#             with st.spinner("Retrieving relevant chunks..."):
#                 chunks = st.session_state.retriever.retrieve(query, top_k=top_k)
#             with st.spinner("Generating answer..."):
#                 answer = generate_answer(query, chunks, model_name=config['llm']['model'])
#             st.session_state.chat_history.append((query, answer))
#         else:
#             st.error("Please preprocess documents before asking questions.")
#     else:
#         st.warning("Please enter a question to proceed.")

# # Display chat history
# st.subheader("💬 Chat History")
# for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
#     message(f"**You:** {q}", is_user=True, key=f"user_{i}")
#     message(f"**AI:** {a}", is_user=False, key=f"ai_{i}")

# # Clear history
# if st.button("🗑️ Clear Chat History"):
#     st.session_state.chat_history = []
#     st.success("Chat history cleared.")


# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from pathlib import Path
# import streamlit as st
# from app.retriever import Retriever
# from app.generator import generate_answer
# from utils.config_loader import load_config

# # Environment settings
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# # Page config
# st.set_page_config(page_title="📄 Offline Document Q&A", layout="wide")

# # Sidebar
# with st.sidebar:
#     st.title("⚙️ Settings")
#     config = load_config('config.yaml')

#     # Upload PDFs
#     st.markdown("### 📂 Upload Documents")
#     uploaded_files = st.file_uploader("Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True)
#     if st.button("🧾 Process Documents"):
#         if uploaded_files:
#             os.makedirs("docs", exist_ok=True)
#             with st.spinner("Processing uploaded documents..."):
#                 for file in uploaded_files:
#                     with open(os.path.join("docs", file.name), "wb") as f:
#                         f.write(file.read())
#                 st.success("Files saved and ready for retrieval!")
#         else:
#             st.warning("Please upload at least one document.")

#     st.markdown("---")
#     st.markdown("### 🔍 Retrieval Settings")
#     top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5)

#     st.markdown("---")
#     st.markdown("### 🎨 Theme")
#     theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)
#     if theme == "Dark":
#         st.markdown("""
#             <style>
#             body { background-color: #111; color: #eee; }
#             .stButton>button { background-color: #333; color: white; }
#             </style>
#         """, unsafe_allow_html=True)

#     st.markdown("---")
#     st.markdown("### 🤖 Model Info")
#     st.markdown(f"**Embedding Model:** `{config['embedding']['model']}`")
#     st.markdown(f"**LLM:** `{config['llm']['model']}`")
#     rel_model = config['relevance_model'].get('path', 'None')
#     st.markdown(f"**Relevance Model:** `{rel_model}`")

# # Main Chat UI
# st.title("📚 Ask Your Documents (Offline RAG with LLaMA3)")

# # Initialize retriever once
# @st.cache_resource
# def load_retriever():
#     metadata_dir = os.path.join("docs", "metadata")  # adjust the path as needed
#     return Retriever(
#         metadata_dir,
#         config['embedding']['model'],
#         config['faiss']['index_path'],
#         config['relevance_model'].get('path')
#     )


# retriever = load_retriever()

# # Session state for chat
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Render chat history
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # User input via chat input
# user_input = st.chat_input("Ask a question from your documents...")

# if user_input:
#     # Show user message
#     st.chat_message("user").markdown(user_input)
#     st.session_state.chat_history.append({"role": "user", "content": user_input})

#     # Retrieve & generate answer
#     with st.spinner("Retrieving and generating answer..."):
#         retrieved_chunks = retriever.retrieve(user_input, top_k=top_k)
#         response = generate_answer(user_input, retrieved_chunks, model_name=config['llm']['model'])

#     # Show assistant message
#     st.chat_message("assistant").markdown(response)
#     st.session_state.chat_history.append({"role": "assistant", "content": response})

# # Clear history
# st.markdown("---")
# if st.button("🗑️ Clear Chat History"):
#     st.session_state.chat_history = []
#     st.success("Chat history cleared.")
