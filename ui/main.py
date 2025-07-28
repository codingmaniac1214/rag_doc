# # working


# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from pathlib import Path
# import streamlit as st
# from app.retriever import Retriever
# from app.generator import generate_answer
# from utils.config_loader import load_config
# from streamlit_chat import message

# # Environment & Path Setup
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# # Streamlit Config
# st.set_page_config(page_title="📄 Offline Document Q&A", layout="wide")

# # Sidebar: Load config and file uploader
# with st.sidebar:
#     st.title("⚙️ Settings")
#     config = load_config('config.yaml')

#     st.markdown("### 📂 Upload Documents")
#     uploaded_files = st.file_uploader("Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True)
#     if st.button("🧾 Process Documents"):
#         if uploaded_files:
#             with st.spinner("Processing uploaded documents..."):
#                 for file in uploaded_files:
#                     with open(os.path.join("docs", file.name), "wb") as f:
#                         f.write(file.read())
#                 st.success("Files saved and ready for retrieval!")
#         else:
#             st.warning("Please upload at least one document.")

#     st.markdown("---")
#     st.markdown("### 🔍 Retrieval Settings")
#     top_k = st.slider("Top-K Chunks", min_value=1, max_value=100, value=10)

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

# retriever = Retriever(
#     config['embedding']['model'],                # 2nd: embedding model
#     config['faiss']['index_path'],               # 3rd: faiss index path
#     config['relevance_model'].get('path')        # 4th: optional relevance model
# )

# # Maintain session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Query Input and Answer Generation (First Block)
# query = st.text_input("Enter your question:")
# if st.button("Ask"):
#     if query:
#         with st.spinner("Retrieving relevant chunks..."):
#             chunks = retriever.retrieve(query, top_k=top_k)
#         with st.spinner("Generating answer..."):
#             answer = generate_answer(query, chunks, model_name=config['llm']['model'])
#         st.write("**Answer:**")
#         st.write(answer)

#         # Save to chat history
#         st.session_state.chat_history.append((query, answer))
#     else:
#         st.write("Please enter a question.")

# # Display chat history
# st.subheader("💬 Chat History")
# for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
#     message(f"**You:** {q}", is_user=True, key=f"user_{i}")
#     message(f"**AI:** {a}", is_user=False, key=f"ai_{i}")

# # Clear history
# if st.button("🗑️ Clear Chat History"):
#     st.session_state.chat_history = []
#     st.success("Chat history cleared.")


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
st.set_page_config(page_title="🧠 Neuronyx - Offline Document Q&A", layout="wide")

# Sidebar: Load config and file uploader
with st.sidebar:
    st.title("🧠 Neuronyx Settings")
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
    st.markdown("### 🎨 Theme")
    theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)

# Apply premium theme styles
if theme == "Dark":
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
            color: #e6e6e6;
        }
        .stApp {
            background-color: #0e1117;
            color: #e6e6e6;
        }
        .stButton>button {
            background-color: #333;
            color: white;
            border: 1px solid #666;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            background-color: #222;
            color: white;
            border: 1px solid #444;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #f9f9fb;
            color: #1e1e2f;
        }
        .stButton>button {
            background-color: #4b7bec;
            color: white;
            border: none;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #1e1e2f;
            border: 1px solid #ccc;
        }
        </style>
    """, unsafe_allow_html=True)

# Main UI
st.title("📚 Ask Your Documents - Neuronyx")

retriever = Retriever(
    config['embedding']['model'],                # 2nd: embedding model
    config['faiss']['index_path'],               # 3rd: faiss index path
    config['relevance_model'].get('path')        # 4th: optional relevance model
)

# Maintain session state (changed)
if "chat_history" not in st.session_state or st.session_state.get("force_clear", False):
    st.session_state.chat_history = []
    st.session_state.force_clear = False

# Fixed chunk count
top_k = 10

# Query Input and Answer Generation
query = st.text_input("Enter your question:")
if st.button("Ask"):
    if query:
        with st.spinner("Retrieving relevant chunks..."):
            chunks = retriever.retrieve(query, top_k=top_k)
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, chunks, model_name=config['llm']['model'])
        st.markdown("### 🧠 Answer")
        st.markdown(f"<div style='background-color:#f0f2f6; padding:1rem; border-radius:8px;'>{answer}</div>", unsafe_allow_html=True)

        with st.expander("📄 Show Retrieved Chunks"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"""
                <div style="background-color:#ffffffdd; padding:0.75rem; border-left:4px solid #4b7bec; margin-bottom:1rem; border-radius:6px;">
                <b>Chunk {i+1}:</b><br>{chunk}
                </div>
                """, unsafe_allow_html=True)

        # Save to chat history
        st.session_state.chat_history.append((query, answer))
    else:
        st.write("Please enter a question.")

# Display chat history
st.subheader("💬 Chat History")

chat_style = """
<style>
.chat-bubble-user {
    background-color: #4b7bec22;
    padding: 1rem;
    border-left: 4px solid #4b7bec;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.chat-bubble-ai {
    background-color: #26de8122;
    padding: 1rem;
    border-left: 4px solid #26de81;
    border-radius: 10px;
    margin-bottom: 1rem;
}
</style>
"""
st.markdown(chat_style, unsafe_allow_html=True)

for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
    st.markdown(f'<div class="chat-bubble-user">👤 <b>You:</b><br>{q}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>Neuronyx:</b><br>{a}</div>', unsafe_allow_html=True)

# Clear chat history
if st.button("🗑️ Clear Chat History"):
    st.session_state.force_clear = True
    st.success("Chat history cleared.")
    st.query_params.clear()  # Triggers rerun
