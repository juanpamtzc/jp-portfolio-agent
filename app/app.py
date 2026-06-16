import streamlit as st

# 1. Page Configuration (Must be the very first Streamlit command)
st.set_page_config(
    page_title="JP Portfolio Intelligence Matrix",
    page_icon="🤖",
    layout="wide",  # Wide layout is perfect for side-by-side comparison
    initial_sidebar_state="expanded"
)

# 2. Flawless State Management Initialization
if "messages_generic" not in st.session_state:
    st.session_state.messages_generic = [
        {"role": "assistant", "content": "I am a generic LLM. I have general knowledge up to my cutoff date, but I don't know who JP is."}
    ]

if "messages_specialist" not in st.session_state:
    st.session_state.messages_specialist = [
        {"role": "assistant", "content": "I am the JP Portfolio Specialist Agent. I have real-time access to JP's resume, publications, and codebase metrics via RAG tools."}
    ]

# 3. Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    st.markdown(
        """
        ### App Metrics
        This production layout uses decoupled execution paths to evaluate a baseline model against a context-augmented agent.
        """
    )
    # Placeholder for clearing history if needed
    if st.button("Reset Chat Histories"):
        st.session_state.messages_generic = [st.session_state.messages_generic[0]]
        st.session_state.messages_specialist = [st.session_state.messages_specialist[0]]
        st.rerun()

# 4. Main App Layout UI
st.title("🤖 JP Portfolio Intelligence & Evaluation Engine")
st.caption("A production-grade demonstration of RAG-driven Agentic Orchestration vs. Foundation Models.")

# Create the side-by-side comparison column grid
col_generic, col_specialist = st.columns(2)

# --- Left Column: Generic Foundation Model Interface ---
with col_generic:
    st.subheader("🌐 Baseline Foundation LLM")
    st.info("Direct inference pipeline without custom domain visibility.")
    
    # Render chat history from state
    for msg in st.session_state.messages_generic:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# --- Right Column: JP-Specialist Agent Interface ---
with col_specialist:
    st.subheader("⚡ JP Specialist Agent (RAG + Tools)")
    st.success("Orchestrated Agent with deep context retrieval capabilities.")
    
    # Render chat history from state
    for msg in st.session_state.messages_specialist:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# 5. Core Unified Chat Input Axis
user_query = st.chat_input("Ask a professional question (e.g., 'What projects has JP built using PyTorch?')")

if user_query:
    # Append the user's message to BOTH histories to keep the evaluation perfectly paired
    st.session_state.messages_generic.append({"role": "user", "content": user_query})
    st.session_state.messages_specialist.append({"role": "user", "content": user_query})
    
    # Force immediate UI rerun to show the user's typed question before generating responses
    st.rerun()