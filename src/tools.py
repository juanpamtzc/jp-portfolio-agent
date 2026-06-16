# src/tools.py
import json
import pandas as pd
from scipy.spatial.distance import cdist

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
        },
        {
            "name": "find_player_replacement",
            "description": "Calculates live statistical player replacements for a given football player using the Tactical Scouting Engine. Triggers when the user asks 'who plays like', 'replacement for', or mentions players like Messi or Iniesta.",
            "parameters": {"query": "string"}
        }
    ]

def find_player_replacement(query: str) -> str:
    """
    Pulls data directly from the deployed Tactical Scouting Engine repository,
    runs the PCA Euclidean distance math, and returns the top 3 replacements.
    """
    # 1. Raw GitHub URLs (No local files needed!)
    # Note: Replace 'juanpamtzc' if your actual GitHub username differs in the URL
    pca_url = "https://raw.githubusercontent.com/juanpamtzc/futbol-id/main/data/pca_centroids.csv"
    meta_url = "https://raw.githubusercontent.com/juanpamtzc/futbol-id/main/data/model_metadata.csv"
    
    try:
        # 2. Load data directly into memory from the cloud
        pca_df = pd.read_csv(pca_url, index_col="Player")
        metadata = pd.read_csv(meta_url, index_col="Model")
        
        # 3. Intelligent Name Matching
        # Searches the query to find a matching player name from the database
        query_lower = query.lower()
        exact_player = None
        
        # We loop through the index to see if the user typed a name like "iniesta"
        for player_name in pca_df.index:
            last_name = player_name.lower().split()[-1]
            if last_name in query_lower or player_name.lower() in query_lower:
                exact_player = player_name
                break
                
        if not exact_player:
            return "Could not identify a player from the 2015-16 La Liga database in the query. Please provide a specific name."
            
        # 4. Run the exact math from your scouting engine!
        target_vector = pca_df.loc[exact_player].values.reshape(1, -1)
        distances = cdist(target_vector, pca_df.values, metric='euclidean').flatten()
        
        baseline_dist = metadata.loc["PCA", "Self_Distance_Baseline"]
        norm_distances = distances / baseline_dist
        
        # 5. Sort and filter
        df_results = pd.DataFrame({
            'Player': pca_df.index,
            'Multiplier': norm_distances
        }).sort_values('Multiplier')
        
        # Get top 3 (excluding the player themselves)
        df_replacements = df_results[df_results['Player'] != exact_player].head(3)
        
        # 6. Format the JSON/String for the Groq LLM to read
        result_str = f"Tactical Scouting Engine Results for '{exact_player}':\n"
        for i, row in df_replacements.iterrows():
            result_str += f"- {row['Player']} (Tactical Variance Multiplier: {row['Multiplier']:.2f}x)\n"
            
        return result_str
        
    except Exception as e:
        return f"Tool Execution Error: Failed to fetch or process data. Details: {str(e)}"