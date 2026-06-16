# srcwell/model_pipeline.py
# src/model_pipeline.py
import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from huggingface_hub import InferenceClient

@st.cache_resource(show_spinner=False)
def initialize_rag_index(data_dir: str = "data"):
    """
    Ingests professional profile text files, runs local embedding tokenization,
    and constructs the VectorStoreIndex. 
    Cached to prevent massive CPU/GPU memory re-allocations on page re-runs.
    """
    # Defensive check: Ensure data directory exists and contains data
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        return None
    
    # 1. Read files from your /data folder (Notebook 3 practice)
    reader = SimpleDirectoryReader(data_dir)
    documents = reader.load_data()
    
    # 2. Cache the embedding model locally on la-maquina for local testing, 
    # or let Streamlit Cloud spin it up in its own memory container.
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 3. Assemble local context matrix
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    return index

@st.cache_resource(show_spinner=False)
def get_hf_client():
    """
    Initializes the production Hugging Face Inference client.
    Pulls the API token securely from streamlit secrets or environment vars.
    """
    # Streamlit looks for secrets in .streamlit/secrets.toml locally, or the Cloud Dashboard settings
    hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    
    if not hf_token:
        st.warning("⚠️ Hugging Face Token ('HF_TOKEN') missing. Falling back to unauthenticated public tier.")
    
    return InferenceClient(token=hf_token)

def run_baseline_inference(prompt: str) -> str:
    """
    Queries a generic base model (e.g., standard Meta-Llama-3-8B-Instruct) 
    via Hugging Face serverless API to simulate a zero-context foundation model.
    """
    client = get_hf_client()
    try:
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Baseline API Error: {str(e)}"

def run_specialist_agent_inference(prompt: str, index) -> tuple:
    """
    Executes RAG context retrieval, constructs an augmented prompt,
    and invokes your custom fine-tuned model hosted on Hugging Face.
    Returns a tuple: (Synthesized Response, List of Retrieved Context Blocks)
    """
    retrieved_contexts = []
    
    # 1. Pull hyper-relevant career context blocks if RAG is ready
    if index:
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(prompt)
        retrieved_contexts = [node.text for node in nodes]
    
    # 2. Build the System Directive & Context Prompt injection (Notebook 3 Strategy)
    context_str = "\n---\n".join(retrieved_contexts) if retrieved_contexts else "No context found."
    
    system_prompt = (
        "You are the JP Portfolio Specialist Agent, a fine-tuned expert on JP's career. "
        "Use the provided context to answer the user's inquiry accurately and professionally."
    )
    
    user_augmented_prompt = f"Context:\n{context_str}\n\nQuery:\n{prompt}"
    
    # 3. Query your custom fine-tuned repo weights on Hugging Face
    client = get_hf_client()
    try:
        # REPLACE 'your-username/your-fine-tuned-model' with your actual HF repo name later
        repo_id = st.secrets.get("HF_MODEL_REPO", "google/gemma-2-2b-it") 
        
        response = client.chat_completion(
            model=repo_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_augmented_prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content, retrieved_contexts
    except Exception as e:
        return f"Specialist API Error: {str(e)}", retrieved_contexts