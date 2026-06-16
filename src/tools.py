# src/tools.py
import json

# =====================================================================
# CORE PORTFOLIO DATA (Hardcoded Source of Truth for GitHub/Cloud Deployment)
# =====================================================================
PORTFOLIO_DATA = {
    "profile": {
        "name": "Juan Pablo Martínez Cordeiro, Ph.D. (JP)",
        "role": "Computational Scientist & Mechanical Engineer",
        "education": "Ph.D. from UT Austin",
        "bio": "Specializes in bridging the gap between mathematical/statistical modeling and high-performance code, focusing on physics-backed data pipelines, fluid dynamics, and scalable computational models.",
        "skills": {
            "Engineering": ["Fluid Dynamics", "Heat Transfer", "Thermodynamics/Stat Mech", "Multiscale and Nonlinear FEA/CFD", "Nanofluidics/Nanomechanics (MD/DFT)"],
            "Mathematics": ["Numerical Methods", "Stochastic Modeling", "Statistics", "Probability", "Linear Algebra"],
            "Programming": ["High-Performance Computing", "Python", "C", "MATLAB", "SQL", "Vectorization", "Parallelization (CUDA)", "CI/CD", "Machine Learning"],
            "Analysis": ["Statistical Inference", "Hypothesis Testing", "Model Development/Validation", "Time-series Analysis", "Spectral Analysis"]
        }
    },
    "completed_projects": [
        {
            "id": "blue_vs_red",
            "name": "🔴 🔵 Blue vs. Red: Expected Utility Optimizer",
            "description": "An interactive dashboard computing expected utility under existential risk, balancing self-preservation against the greater good.",
            "tech_stack": "Streamlit, Computational Probability, Expected Utility Theory",
            "url": "https://red-vs-blue.streamlit.app"
        },
        {
            "id": "tactical_scouting",
            "name": "⚽ Tactical Scouting Engine: Latent-Space Player Replacement",
            "description": "Compresses professional football player performance profiles into a latent space to identify mathematically accurate, style-matched replacements.",
            "tech_stack": "Python, Machine Learning, Latent Space Dimensionality Reduction, Streamlit",
            "url": "https://futbol-id.streamlit.app"
        }
    ],
    "upcoming_projects": [
        {
            "id": "openfoam_surrogate",
            "name": "🌡️ OpenFOAM Thermal Surrogate Model",
            "status": "Running CFD Simulations to Train Surrogate Model",
            "challenge": "Traditional geometric optimization of heat dissipation fins requires computationally expensive fluid dynamics iterations in OpenFOAM.",
            "approach": "Training a data-driven surrogate model on a physics-backed dataset to predict thermal profiles in milliseconds."
        },
        {
            "id": "telemetry_forecasting",
            "name": "🏎️ High-Frequency Telemetry Forecasting",
            "status": "Designing Pipeline to Sync Data",
            "challenge": "F1 cars broadcast asynchronous, noisy telemetry across independent channels (Speed, Throttle, Brake, RPM).",
            "approach": "Engineering a synchronized time-series pipeline to feed ML models, enabling real-time vehicle state forecasting and instant detection of anomalies like tire lock-ups."
        },
        {
            "id": "adverse_selection",
            "name": "🍋 Simulating Adverse Selection",
            "status": "Finalizing Problem Statement",
            "challenge": "In markets with asymmetric information, generic or uninformative priors mathematically guarantee expected losses for the buyer over time.",
            "approach": "Building a computational probability engine in Python to map market loss distributions and create interactive visualizations of shifting risk surfaces under informed priors."
        }
    ]
}

# =====================================================================
# EXPOSED AGENT TOOLS
# =====================================================================

def get_profile_summary(query: str = "") -> str:
    """Retrieves JP's core skills, background, education, and Ph.D. competencies."""
    p = PORTFOLIO_DATA["profile"]
    return json.dumps({
        "name": p["name"],
        "role": p["role"],
        "education": p["education"],
        "skills": p["skills"]
    }, indent=2)


def get_project_details(project_keywords: str) -> str:
    """
    Searches and retrieves metrics, challenges, tech stacks, status, 
    and links for projects matching keywords (e.g., 'OpenFOAM', 'F1', 'Scouting', 'Red').
    """
    kw = project_keywords.lower()
    results = []
    
    # Check completed
    for proj in PORTFOLIO_DATA["completed_projects"]:
        if kw in proj["name"].lower() or kw in proj["description"].lower() or kw in proj["id"]:
            results.append({"Type": "Deployed Production Project", **proj})
            
    # Check upcoming
    for proj in PORTFOLIO_DATA["upcoming_projects"]:
        if kw in proj["name"].lower() or kw in proj["challenge"].lower() or kw in proj["approach"].lower() or kw in proj["id"]:
            results.append({"Type": "In-Development Surrogate/Simulation", **proj})
            
    if not results:
        return f"No specific project match found for '{project_keywords}'. Available profiles cover: Blue vs Red, Tactical Scouting Engine, OpenFOAM Thermal Surrogate, High-Frequency Telemetry Forecasting, and Simulating Adverse Selection."
        
    return json.dumps(results, indent=2)


def generate_tool_descriptions() -> list:
    """Generates the JSON schema mapping available execution endpoints for the LLM core."""
    return [
        {
            "name": "get_profile_summary",
            "description": "Retrieves JP's academic background, Ph.D. credentials, and categorized technical skill matrices.",
            "parameters": {"query": "string"}
        },
        {
            "name": "get_project_details",
            "description": "Retrieves comprehensive specs, status updates, deployment links, and engineering approaches for specific portfolio items via keywords like 'OpenFOAM', 'F1', 'Scouting', 'Red'.",
            "parameters": {"project_keywords": "string"}
        }
    ]