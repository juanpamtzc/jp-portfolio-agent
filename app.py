# app.py
import streamlit as st

# Clean, native Python imports!
from src.model_pipeline import initialize_rag_index, run_baseline_inference, run_specialist_agent_inference
from src.tools import get_profile_summary, get_project_details, find_player_replacement

# 1. Page Configuration
st.set_page_config(
    page_title="JP Portfolio AI LLM Agent",
    layout="wide",
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

# 4. Sidebar Control Panel
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    st.markdown("### Evaluation Monitor")
    st.caption("This system demonstrates cross-pipeline quality evaluation.")
    if st.button("Reset Chat Histories"):
        st.session_state.messages_generic = [st.session_state.messages_generic[0]]
        st.session_state.messages_specialist = [st.session_state.messages_specialist[0]]
        st.rerun()

# 5. UI Layout
st.title("🤖 JP Portfolio AI LLM Agent")
st.caption("A production-grade demonstration of Foundation LLM Models vs. RAG-driven Agentic Orchestration.")

col_generic, col_specialist = st.columns(2)

with col_generic:
    st.subheader("🌐 Baseline Foundation LLM")
    st.info("Direct API inference without custom domain visibility.")
    for msg in st.session_state.messages_generic:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

with col_specialist:
    st.subheader("⚡ JP Specialist Agent (RAG + Tools)")
    st.success("Orchestrated Agent with deep context retrieval capabilities.")
    for msg in st.session_state.messages_specialist:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("context"):
                with st.expander("📚 Retained Source Context"):
                    for doc in msg["context"]:
                        st.caption(doc)

# 6. Chat Input Logic
if user_query := st.chat_input("Ask about JP's engineering experience or projects..."):
    
    # Append user prompt immediately
    st.session_state.messages_generic.append({"role": "user", "content": user_query})
    st.session_state.messages_specialist.append({"role": "user", "content": user_query})
    
    with col_generic:
        with st.chat_message("user"):
            st.markdown(user_query)
    with col_specialist:
        with st.chat_message("user"):
            st.markdown(user_query)
            
    # Run Baseline Model
    with col_generic:
        with st.chat_message("assistant"):
            with st.spinner("Generating baseline output..."):
                baseline_res = run_baseline_inference(user_query)
            st.markdown(baseline_res)
            st.session_state.messages_generic.append({"role": "assistant", "content": baseline_res})
            
    # Run Agentic Specialist Model
    with col_specialist:
        with st.chat_message("assistant"):
            with st.status("Agent thinking...", expanded=True) as status:
                
                eval_input = user_query
                tool_output = None
                tool_name = None
                
                # Intelligent routing simulation over our improved tool list
                query_lower = user_query.lower()
                
                # Route 1: Profile & Resume Data
                if any(word in query_lower for word in ["resume", "job", "experience", "skill", "education", "phd", "who is"]):
                    tool_name = "get_profile_summary"
                    status.write(f"⚙️ Tool Match Detected: Forwarding to `{tool_name}`...")
                    tool_output = get_profile_summary()
                
                # Route 2: Project Specifics
                elif any(word in query_lower for word in ["project", "openfoam", "f1", "telemetry", "scout", "red", "blue", "adverse", "portfolio"]):
                    tool_name = "get_project_details"
                    status.write(f"⚙️ Tool Match Detected: Forwarding to `{tool_name}`...")
                    tool_output = get_project_details(user_query)
                
                # Route 3: Live Tactical Scouting Math Engine
                elif any(word in query_lower for word in ["replace", "scout", "plays like", "iniesta", "messi", "player", "football", "soccer"]):
                    tool_name = "find_player_replacement"
                    status.write(f"⚙️ Live Engine Matched: Executing `{tool_name}` via GitHub Raw Data...")
                    # Pass the entire user query to the tool so it can extract the name
                    tool_output = find_player_replacement(user_query)

                # Inject JSON tool output into the LLM prompt if a tool fired
                if tool_output:
                    with st.expander(f"🛠️ Executed Tool: {tool_name}()", expanded=True):
                        st.code(tool_output, language="json")
                    eval_input += f"\n\n[Supplemental Structured Tool Data]:\n{tool_output}"
                
                status.write("🧠 Querying LlamaIndex RAG and Groq weights...")
                specialist_res, context_blocks = run_specialist_agent_inference(eval_input, index)
                status.update(label="Inference Complete!", state="complete", expanded=False)
                
            st.markdown(specialist_res)
            
            if context_blocks:
                with st.expander("📚 RAG Source Context Retained"):
                    for block in context_blocks:
                        st.caption(block)
                        
            st.session_state.messages_specialist.append({
                "role": "assistant", 
                "content": specialist_res,
                "context": context_blocks
            })