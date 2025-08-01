# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from pathlib import Path
# import streamlit as st
# from app.retriever import Retriever
# from app.generator import generate_answer
# from utils.config_loader import load_config
# from streamlit_chat import message

# from scripts import run_preprocessing

# # Environment & Path Setup
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# # Streamlit Config
# st.set_page_config(page_title="🧠 Neuronyx - Offline Document Q&A", layout="wide")

# # Sidebar: Load config and file uploader
# with st.sidebar:
#     st.title("🧠 Neuronyx Settings")
#     config = load_config('config.yaml')

#     st.markdown("### 📂 Upload Documents")
#     uploaded_files = st.file_uploader("Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True)
#     if st.button("🧾 Process Documents"):
#         if uploaded_files:
#             with st.spinner("Processing uploaded documents..."):
#                 os.makedirs("data/raw_pdfs",exist_ok=True)
#                 for file in uploaded_files:
#                     file_path=os.path.join("data/raw_pdfs",file.name)
#                     with open(file_path, "wb") as f:
#                         f.write(file.read())
#                 run_preprocessing.main()

#                 st.success("Files saved and ready for retrieval!")
#         else:
#             st.warning("Please upload at least one document.")

#     st.markdown("---")
#     st.markdown("### 🎨 Theme")
#     theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)

# # Apply premium theme styles
# if theme == "Dark":
#     st.markdown("""
#         <style>
#         body {
#             background-color: #0e1117;
#             color: #e6e6e6;
#         }
#         .stApp {
#             background-color: #0e1117;
#             color: #e6e6e6;
#         }
#         .stButton>button {
#             background-color: #333;
#             color: white;
#             border: 1px solid #666;
#             border-radius: 8px;
#         }
#         .stTextInput>div>div>input {
#             background-color: #222;
#             color: white;
#             border: 1px solid #444;
#         }
#         </style>
#     """, unsafe_allow_html=True)
# else:
#     st.markdown("""
#         <style>
#         .stApp {
#             background-color: #f9f9fb;
#             color: #1e1e2f;
#         }
#         .stButton>button {
#             background-color: #4b7bec;
#             color: white;
#             border: none;
#             border-radius: 8px;
#         }
#         .stTextInput>div>div>input {
#             background-color: #ffffff;
#             color: #1e1e2f;
#             border: 1px solid #ccc;
#         }
#         </style>
#     """, unsafe_allow_html=True)

# # Main UI
# st.title("📚 Ask Your Documents - Neuronyx")

# retriever = Retriever(
#     config['embedding']['model'],                # 2nd: embedding model
#     config['faiss']['index_path'],               # 3rd: faiss index path
#     config['relevance_model'].get('path')        # 4th: optional relevance model
# )

# # Maintain session state (changed)
# if "chat_history" not in st.session_state or st.session_state.get("force_clear", False):
#     st.session_state.chat_history = []
#     st.session_state.force_clear = False

# # Fixed chunk count
# top_k = 10

# # Query Input and Answer Generation
# query = st.text_input("Enter your question:")
# if st.button("Ask"):
#     if query:
#         with st.spinner("Retrieving relevant chunks..."):
#             chunks = retriever.retrieve(query, top_k=top_k)
#         with st.spinner("Generating answer..."):
#             answer = generate_answer(query, chunks, model_name=config['llm']['model'])
#         st.markdown("### 🧠 Answer")
#         st.markdown(f"<div style='background-color:#f0f2f6; padding:1rem; border-radius:8px;'>{answer}</div>", unsafe_allow_html=True)

#         with st.expander("📄 Show Retrieved Chunks"):
#             for i, chunk in enumerate(chunks):
#                 st.markdown(f"""
#                 <div style="background-color:#ffffffdd; padding:0.75rem; border-left:4px solid #4b7bec; margin-bottom:1rem; border-radius:6px;">
#                 <b>Chunk {i+1}:</b><br>{chunk}
#                 </div>
#                 """, unsafe_allow_html=True)

#         # Save to chat history
#         st.session_state.chat_history.append((query, answer))
#     else:
#         st.write("Please enter a question.")

# # Display chat history
# st.subheader("💬 Chat History")

# chat_style = """
# <style>
# .chat-bubble-user {
#     background-color: #4b7bec22;
#     padding: 1rem;
#     border-left: 4px solid #4b7bec;
#     border-radius: 10px;
#     margin-bottom: 1rem;
# }
# .chat-bubble-ai {
#     background-color: #26de8122;
#     padding: 1rem;
#     border-left: 4px solid #26de81;
#     border-radius: 10px;
#     margin-bottom: 1rem;
# }
# </style>
# """
# st.markdown(chat_style, unsafe_allow_html=True)

# for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
#     st.markdown(f'<div class="chat-bubble-user">👤 <b>You:</b><br>{q}</div>', unsafe_allow_html=True)
#     st.markdown(f'<div class="chat-bubble-ai">🤖 <b>Neuronyx:</b><br>{a}</div>', unsafe_allow_html=True)

# # Clear chat history
# if st.button("🗑️ Clear Chat History"):
#     st.session_state.force_clear = True
#     st.success("Chat history cleared.")
#     st.query_params.clear()  # Triggers rerun


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
import streamlit as st
from app.retriever import Retriever
from app.generator import generate_answer
from utils.config_loader import load_config
from streamlit_chat import message
from scripts import run_preprocessing
import json
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", filename="log.log")


# Environment & Path Setup
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Streamlit Config
st.set_page_config(page_title="🧠 Neuronyx - Offline Document Q&A", layout="wide")


# Load or initialize products
PRODUCTS_FILE = "products.json"
def load_products():
   if os.path.exists(PRODUCTS_FILE):
       with open(PRODUCTS_FILE, "r") as f:
           return json.load(f).get("products", ["default"])
   return ["default"]


def save_products(products):
   with open(PRODUCTS_FILE, "w") as f:
       json.dump({"products": products}, f)
   logging.debug(f"Saved products to {PRODUCTS_FILE}")


# Initialize session state
if "chat_history" not in st.session_state or st.session_state.get("force_clear", False):
   st.session_state.chat_history = []
   st.session_state.force_clear = False
if "products" not in st.session_state:
   st.session_state.products = load_products()
if "selected_product" not in st.session_state:
   st.session_state.selected_product = st.session_state.products[0]


# Sidebar: Load config and file uploader
with st.sidebar:
   st.title("🧠 Neuronyx Settings")
   config = load_config('config.yaml')


   # Product selection
   st.markdown("### 📋 Select Product")
   selected_product = st.selectbox("Choose Product", st.session_state.products, key="product_select")
   if selected_product != st.session_state.selected_product:
       st.session_state.selected_product = selected_product
       st.session_state.retriever = None  # Reset retriever on product change
       logging.debug(f"Switched to product '{st.session_state.selected_product}'")


   # Create new product
   st.markdown("### ➕ Create New Product")
   new_product = st.text_input("Enter Product Name")
   if st.button("Add Product"):
       if new_product and new_product not in st.session_state.products:
           # Create product directories
           data_dirs = [
               os.path.join(config['base_data_dir'], new_product, dir_name)
               for dir_name in ["raw_pdfs", "cleaned_texts", "metadata", "chunks"]
           ]
           embeddings_dir = os.path.join(config['base_embeddings_dir'], new_product)
           try:
               for dir_path in data_dirs + [embeddings_dir]:
                   os.makedirs(dir_path, exist_ok=True)
                   logging.debug(f"Created directory: {dir_path}")
               st.session_state.products.append(new_product)
               save_products(st.session_state.products)
               st.session_state.selected_product = new_product
               st.session_state.retriever = None  # Reset retriever for new product
               st.success(f"Created product '{new_product}' with directories. Upload PDFs to start.")
               logging.info(f"Successfully created product '{new_product}'")
           except Exception as e:
               st.error(f"Failed to create product '{new_product}': {str(e)}")
               logging.error(f"Failed to create directories for product '{new_product}': {str(e)}")
       elif not new_product:
           st.warning("Please enter a product name.")
       else:
           st.warning(f"Product '{new_product}' already exists.")


   st.markdown("### 📂 Upload Documents")
   uploaded_files = st.file_uploader(f"Upload PDFs or Text Files for {st.session_state.selected_product}", type=["pdf", "txt"], accept_multiple_files=True)
   if st.button("🧾 Process Documents"):
       if uploaded_files:
           with st.spinner("Processing uploaded documents..."):
               # Save uploaded files to data/<product>/raw_pdfs
               raw_pdfs_dir = os.path.join(config['base_data_dir'], st.session_state.selected_product, "raw_pdfs")
               os.makedirs(raw_pdfs_dir, exist_ok=True)
               for file in uploaded_files:
                   file_path = os.path.join(raw_pdfs_dir, file.name)
                   with open(file_path, "wb") as f:
                       f.write(file.read())
                   logging.debug(f"Saved uploaded file to {file_path}")
              
               # Call preprocessing pipeline
               try:
                   run_preprocessing.main(st.session_state.selected_product)
                   st.success("Files uploaded and preprocessed successfully! ✅")
                   # Reinitialize retriever after preprocessing
                   index_path = os.path.join(config['base_embeddings_dir'], st.session_state.selected_product, "faiss_index")
                   st.session_state.retriever = Retriever(
                       config['embedding']['model'],
                       index_path,
                       config['relevance_model'].get('path')
                   )
                   logging.debug(f"Reinitialized Retriever for product '{st.session_state.selected_product}' after preprocessing")
               except Exception as e:
                   st.error(f"Preprocessing failed: {str(e)}")
                   logging.error(f"Preprocessing failed for product '{st.session_state.selected_product}': {str(e)}")
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


# Initialize retriever for selected product if not already set
if "retriever" not in st.session_state or st.session_state.get("last_product") != st.session_state.selected_product:
   try:
       index_path = os.path.join(config['base_embeddings_dir'], st.session_state.selected_product, "faiss_index")
       st.session_state.retriever = Retriever(
           config['embedding']['model'],
           index_path,
           config['relevance_model'].get('path')
       )
       st.session_state.last_product = st.session_state.selected_product
       logging.debug(f"Initialized Retriever for product '{st.session_state.selected_product}'")
   except Exception as e:
       logging.warning(f"Retriever initialization failed for product '{st.session_state.selected_product}': {e}. Upload and process documents to create the index.")
       st.warning(f"No documents processed for product '{st.session_state.selected_product}'. Please upload and process documents.")
       st.session_state.retriever = None


# Fixed chunk count
top_k = 10


# Query Input and Answer Generation
query = st.text_input(f"Enter your question for {st.session_state.selected_product}:")
if st.button("Ask"):
   if query:
       if st.session_state.retriever is None:
           st.error(f"No documents processed for product '{st.session_state.selected_product}'. Please upload and process documents first.")
           logging.warning(f"Query attempted with no Retriever for product '{st.session_state.selected_product}'")
       else:
           with st.spinner("Retrieving relevant chunks..."):
               chunks = st.session_state.retriever.retrieve(query, top_k=top_k)
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
           logging.debug(f"Generated answer for query: {query}")
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



