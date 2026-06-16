import streamlit as st
import time
# Assuming you run the app from the root directory using: streamlit run app/app.py
from src.model_pipeline import initialize_rag_index, run_baseline_inference, run_specialist_agent_inference
from src.tools import look_up_project_details

# 1. Page Configuration (Must be first)
st.set_page_config(
    page_title="JP Portfolio Intelligence",
    page_icon="🤖",
    layout="wide",  # Crucial for the side-by-side matrix
    initial_sidebar_state="expanded"
)

# 2. Lazy-Load RAG Components (Cached)
with st.spinner("Initializing Vector Index and Context Matrices..."):
    # Pointing to the data directory at the root
    index = initialize_rag_index(data_dir="data")

# 3. Flawless State Management Initialization
if "messages_generic" not in st.session_state:
    st.session_state.messages_generic = [
        {"role": "assistant", "content": "I am a generic Foundation LLM. I do not have access to JP's specific career data or private repositories."}
    ]

if "messages_specialist" not in st.session_state:
    st.session_state.messages_specialist = [
        {"role": "assistant", "content": "I am the JP Portfolio Specialist Agent. I am wired into JP's resume, publications, and codebase metrics via RAG tools."}
    ]

# 4. Main App Header
st.title("🤖 JP Portfolio Intelligence & Evaluation Engine")
st.caption("A production-grade demonstration of RAG-driven Agentic Orchestration vs. Foundation Models.")

# 5. Build the Evaluation Matrix Columns
col_generic, col_specialist = st.columns(2)

with col_generic:
    st.subheader("🌐 Baseline Foundation LLM")
    st.info("Direct API inference without custom domain visibility.")
    # Render historical messages
    for msg in st.session_state.messages_generic:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

with col_specialist:
    st.subheader("⚡ JP Specialist Agent (RAG + Tools)")
    st.success("Orchestrated Agent with deep context retrieval capabilities.")
    # Render historical messages
    for msg in st.session_state.messages_specialist:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render historical RAG context if it exists
            if msg.get("context"):
                with st.expander("📚 Retained Source Context"):
                    for doc in msg["context"]:
                        st.caption(doc)

# 6. Core Unified Chat Input Axis
if user_query := st.chat_input("Ask about JP's engineering experience (e.g., 'What are the metrics for the Waymo project?')."):
    
    # --- UPDATE UI IMMEDIATELY ---
    # Append user prompt to states
    st.session_state.messages_generic.append({"role": "user", "content": user_query})
    st.session_state.messages_specialist.append({"role": "user", "content": user_query})
    
    # Render user prompt in both columns instantly
    with col_generic:
        with st.chat_message("user"):
            st.markdown(user_query)
    with col_specialist:
        with st.chat_message("user"):
            st.markdown(user_query)
            
    # --- EXECUTE INFERENCE IN PARALLEL(ISH) ---
    
    # A. Baseline Execution
    with col_generic:
        with st.chat_message("assistant"):
            with st.spinner("Generating generic response..."):
                baseline_response = run_baseline_inference(user_query)
            st.markdown(baseline_response)
            st.session_state.messages_generic.append({"role": "assistant", "content": baseline_response})
            
    # B. Specialist Execution (With Tool Visibility)
    with col_specialist:
        with st.chat_message("assistant"):
            with st.status("Agent thinking...", expanded=True) as status:
                
                eval_input = user_query
                retrieved_tool_output = None
                
                # Simulate Tool Routing Logic
                if any(word in user_query.lower() for word in ["project", "metric", "waymo", "portfolio"]):
                    status.write("⚙️ Tool Match Detected: Querying `look_up_project_details`...")
                    proj_name = "portfolio" if "portfolio" in user_query.lower() else "waymo"
                    retrieved_tool_output = look_up_project_details(proj_name)
                    
                    with st.expander(f"🛠️ Executed Tool: look_up_project_details('{proj_name}')", expanded=True):
                        st.code(retrieved_tool_output, language="yaml")
                        
                    eval_input += f"\n[Supplemental Tool Metrics]: {retrieved_tool_output}"
                
                status.write("🧠 Pinging Inference API & Vector Database...")
                specialist_response, context_blocks = run_specialist_agent_inference(eval_input, index)
                
                status.update(label="Response Synthesized!", state="complete", expanded=False)
            
            # Print Final Synthesis
            st.markdown(specialist_response)
            
            # Render RAG Context Visibility
            if context_blocks:
                with st.expander("📚 RAG Source Context Retained"):
                    for block in context_blocks:
                        st.caption(block)
                        
            st.session_state.messages_specialist.append({
                "role": "assistant", 
                "content": specialist_response,
                "context": context_blocks
            })