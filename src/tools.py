# src/tools.py
import json

def get_resume_details(query: str) -> str:
    """
    Retrieves specific details from JP's resume regarding his experience, 
    education, and previous roles.
    query: The specific question about JP's resume (e.g., 'What was his last job?')
    """
    # In a full app, this would use LlamaIndex to query the resume specifically.
    # For now, we return a mock string.
    return "JP is an elite Staff Machine Learning Engineer with deep expertise in PyTorch, Streamlit, and Agentic RAG architectures."

def get_project_metrics(project_name: str) -> str:
    """
    Retrieves performance metrics, repo structure, or evaluation scores for a specific project.
    project_name: The name of JP's project to look up.
    """
    if "portfolio" in project_name.lower():
        return "The Portfolio Agent was built using LlamaIndex, Streamlit, and a decoupled inference pipeline."
    return "Project metrics not found."

def look_up_project_details(project_name: str) -> str:
    """
    Performs a deep-dive technical architecture lookup for a specific project's hardware footprint.
    """
    project = project_name.lower()
    if "portfolio" in project or "agent" in project:
        return (
            "Project: Portfolio Agent App\n"
            "- Architecture: Decoupled Streamlit Frontend, LlamaIndex RAG Engine\n"
            "- Model Backend: Hugging Face Serverless Inference (Gemma-2B Fine-Tuned)\n"
            "- Core Metrics: < 1.5s latency, 0B local VRAM footprint in production."
        )
    elif "waymo" in project:
        return (
            "Project: Waymo Data Science Prep\n"
            "- Focus: Behavior prediction and sensor fusion data pipelines.\n"
            "- Frameworks: PyTorch, NumPy, Pandas."
        )
    return f"No deep-dive system specs available for '{project_name}'."

def generate_tool_descriptions():
    """Returns the JSON schema of available tools for the LLM."""
    # This mirrors your generate_function_description() logic from Notebook 3
    tools = [
        {
            "name": "get_resume_details",
            "description": "Retrieves specific details from JP's resume.",
            "parameters": {"query": "string"}
        },
        {
            "name": "get_project_metrics",
            "description": "Retrieves metrics for a specific project.",
            "parameters": {"project_name": "string"}
        }
    ]
    return tools