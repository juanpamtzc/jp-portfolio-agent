# src/model_pipeline.py
import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# 1. Secure Secret Management Access
# Fetches from .streamlit/secrets.toml locally or the Streamlit Cloud dashboard
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

@st.cache_resource(show_spinner=False)
def initialize_rag_index(data_dir: str = "data"):
    """
    Ingests professional profile text files, runs local embedding tokenization,
    and constructs the VectorStoreIndex using Groq as the core LLM driver.
    """
    # Defensive check: Ensure data directory exists and contains data
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        return None
    
    # Configure Groq as the global LLM engine within LlamaIndex
    if not GROQ_API_KEY:
        st.error("❌ Critical Error: 'GROQ_API_KEY' is missing from your Streamlit Secrets.")
        st.stop()
        
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    
    # Configure free, local embedding calculations
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Read files from your /data folder
    reader = SimpleDirectoryReader(data_dir)
    documents = reader.load_data()
    
    # Assemble local context matrix
    index = VectorStoreIndex.from_documents(documents)
    return index

def run_baseline_inference(prompt: str) -> str:
    """
    Queries a generic baseline model directly via Groq cloud API 
    to simulate a zero-context foundation model response.
    """
    if not GROQ_API_KEY:
        return "Baseline Error: GROQ_API_KEY is not configured."
        
    try:
        # Initializing an isolated Groq instance to guarantee zero-context leakage
        llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
        response = llm.complete(prompt)
        return str(response)
    except Exception as e:
        return f"Baseline API Error: {str(e)}"

def run_specialist_agent_inference(prompt: str, index) -> tuple:
    """
    Executes RAG context retrieval, constructs an augmented prompt,
    and invokes the orchestrated Groq model pipeline.
    Returns a tuple: (Synthesized Response, List of Retrieved Context Blocks)
    """
    retrieved_contexts = []
    
    # 1. Pull hyper-relevant career context blocks if RAG is ready
    if index:
        # Retrieve the top 2 closest context blocks matches from your data directory
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(prompt)
        retrieved_contexts = [node.text for node in nodes]
    
    # 2. Build the System Directive & Context Prompt injection
    context_str = "\n---\n".join(retrieved_contexts) if retrieved_contexts else "No context found."
    
    # We pass explicit professional scaffolding to anchor the agent's behavior
    system_directive = (
        "You are the JP Portfolio Specialist Agent, an expert AI representative for JP's career. "
        "Formulate a highly professional response using exclusively the verified context data provided below. "
        "If the answer cannot be derived from the context, state that clearly."
    )
    
    augmented_prompt = (
        f"{system_directive}\n\n"
        f"Verified Context:\n{context_str}\n\n"
        f"User Inquiry:\n{prompt}"
    )
    
    try:
        # Utilize the global LlamaIndex query engine setup
        query_engine = index.as_query_engine(similarity_top_k=2)
        response = query_engine.query(augmented_prompt)
        return str(response), retrieved_contexts
    except Exception as e:
        return f"Specialist API Error: {str(e)}", retrieved_contexts