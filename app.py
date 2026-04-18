import streamlit as st
import pandas as pd
import joblib
import os
import json
from src.ml.train_model import train_and_save_model
from src.agent.graph import create_agent_graph
from src.rag.vectorstore import get_vectorstore
import dotenv

# Load environment variables
dotenv.load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="AI Customer Retention Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
    .report-card {
        background: linear-gradient(145deg, #1e1e24, #2a2a35);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 4px 4px 15px rgba(0,0,0,0.2);
        color: white;
        margin-bottom: 20px;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; font-size: 1.2rem; }
    .risk-medium { color: #faca2b; font-weight: bold; font-size: 1.2rem; }
    .risk-low { color: #00cc96; font-weight: bold; font-size: 1.2rem; }
    
    .action-item {
        background-color: #2b2b36;
        border-left: 5px solid #00cc96;
        padding: 15px;
        margin-top: 10px;
        border-radius: 5px;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #4F46E5 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4338CA !important;
    }
    
    .stAlert { margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- Session State Init ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'graph' not in st.session_state:
    st.session_state.graph = None

# --- Setup Operations ---
@st.cache_resource(show_spinner="Loading ML Model & Vector Store...")
def initialize_system():
    # Make sure model exists
    model_path = "src/ml/model.pkl"
    if not os.path.exists(model_path):
        train_and_save_model("data/customer_churn_sample.csv", model_path)
    model = joblib.load(model_path)
    
    # Initialize VectorDB
    get_vectorstore()
    
    # Initialize LangGraph Agent
    graph = create_agent_graph()
    
    return model, graph

try:
    if "GROQ_API_KEY" not in os.environ:
        st.error("🔑 Please set GROQ_API_KEY in your .env file to enable Groq.")
        st.stop()
        
    model, agent_graph = initialize_system()
    st.session_state.model = model
    st.session_state.graph = agent_graph
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()

# --- Application UI ---

st.title("🛡️ AI Customer Retention Strategy Assistant")
st.markdown("Predict customer churn using Scikit-Learn and generate hyper-personalized agentic retention strategies using LangGraph + ChromaDB RAG.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("👤 Customer Profile")
    st.markdown("Enter customer details to analyze risk and generate a strategy.")
    
    age = st.slider("Age", 18, 90, 45)
    tenure = st.number_input("Tenure (Years)", 0, 10, 3)
    balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 100000.0)
    salary = st.number_input("Estimated Salary ($)", 10000.0, 250000.0, 80000.0)
    num_products = st.selectbox("Num of Products", [1, 2, 3, 4])
    credit_score = st.slider("Credit Score", 300, 850, 650)
    
    col1, col2 = st.columns(2)
    with col1:
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        is_active = st.selectbox("Active Member?", [1, 0], index=1) # Default to 0 for higher churn chance testing
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        has_crcard = st.selectbox("Has Credit Card?", [1, 0])

    analyze_button = st.button("🚀 Analyze Risk & Plan Strategy")

# --- Main Logic ---
if analyze_button:
    customer_data = {
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "EstimatedSalary": salary,
        "NumOfProducts": num_products,
        "CreditScore": credit_score,
        "Geography": geography,
        "IsActiveMember": is_active,
        "Gender": gender,
        "HasCrCard": has_crcard
    }
    
    # 1. Run ML Model
    df_input = pd.DataFrame([customer_data])
    with st.spinner("🤖 Running ML Model..."):
        churn_prob = st.session_state.model.predict_proba(df_input)[0][1]
    
    st.metric(label="Predicted Churn Risk (ML Model)", value=f"{churn_prob:.1%}")
    if churn_prob > 0.5:
        st.error("⚠️ High Risk of Churn Identified.")
    else:
        st.success("✅ Customer seems stable, but let's see what the agent says.")
        
    st.divider()
    
    # 2. Run Agent
    st.subheader("🕵️ Agentic Reasoning & Retention Report")
    
    with st.status("Agent Workflow Running...", expanded=True) as status:
        initial_state = {
            "customer_data": customer_data,
            "churn_score": float(churn_prob),
            "risk_level": "",
            "churn_drivers": [],
            "retrieved_strategies": [],
            "final_recommendations": {},
            "error": ""
        }
        
        st.write("🔍 Running Risk Analyzer Node...")
        # Step through graph (for display purposes, normally you'd just invoke)
        state = initial_state
        for output in st.session_state.graph.stream(initial_state):
            for node_name, state_update in output.items():
                st.write(f"✅ Completed: **{node_name}**")
                state.update(state_update)
                
                if node_name == "RiskAnalyzer":
                    st.write(f"Identified Drivers: {state.get('churn_drivers', [])}")
                elif node_name == "Retriever":
                    st.write(f"Retrieved {len(state.get('retrieved_strategies', []))} relevant strategies from Knowledge Base.")
                    
        status.update(label="Workflow Complete!", state="complete", expanded=False)
        
    if state.get("error"):
        st.error(f"Agent encountered an error: {state['error']}")
        st.json(state['final_recommendations'])
    else:
        report = state["final_recommendations"]
        
        # Display Final Output cleanly
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        
        # Header
        risk_profile = report.get("Risk Profile", {})
        risk_level = risk_profile.get("Risk Level", "Unknown")
        color_class = "risk-high" if "High" in risk_level else ("risk-medium" if "Medium" in risk_level else "risk-low")
        
        st.markdown(f"### 🛡️ Authorized Strategy Report")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Risk Level:** <span class='{color_class}'>{risk_level}</span>", unsafe_allow_html=True)
        col2.markdown(f"**Confidence:** {report.get('Confidence Score', 'Unknown')}")
        col3.markdown(f"**P(Churn):** {risk_profile.get('Churn Probability', 'N/A')}")
        
        # Drivers
        st.markdown("#### 🚨 Key Risk Drivers:")
        for driver in risk_profile.get("Key Drivers", []):
            st.markdown(f"- {driver}")
            
        # Reasoning
        st.markdown("#### 🧠 Agent Reasoning (Why?):")
        st.info(report.get("Reasoning", "No reasoning provided."))
        
        # Actions
        st.markdown("#### 🛠️ Recommended Actions:")
        actions = report.get("Recommended Actions", [])
        for act in actions:
            st.markdown(f"""
            <div class='action-item'>
                <strong>{act.get('Action', 'Action Item')}</strong><br>
                {act.get('Description', '')}
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Expandable Raw JSON for verification
        with st.expander("Show Raw Structured JSON"):
            st.json(report)
