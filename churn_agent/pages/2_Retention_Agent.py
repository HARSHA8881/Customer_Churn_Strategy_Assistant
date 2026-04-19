import streamlit as st
import pandas as pd
import joblib
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup Chroma DB on load if missing
if not os.path.exists("./chroma_db"):
    import subprocess
    subprocess.run(["python", "rag/build_kb.py"])

from agent.graph import create_agent_graph

st.set_page_config(page_title="Retention Strategy Agent", layout="wide")

st.markdown("""
<style>
    body {
        font-family: system-ui, Inter, sans-serif;
        background-color: white;
    }
    .header-main {
        color: #1a1a2e;
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .header-sub {
        color: gray;
        font-size: 14px;
        margin-bottom: 30px;
    }
    .section-header {
        color: #1a1a2e;
        font-weight: 600;
        margin-bottom: 15px;
        margin-top: 25px;
        font-size: 20px;
    }
    .flat-card {
        background-color: #f4f4f4;
        border: 1px solid #dddddd;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .badge-critical, .badge-high {
        background-color: #e63946;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #e9c46a;
        color: #1a1a2e;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #2a9d8f;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    div.stButton > button:first-child {
        background-color: #1a1a2e;
        color: white;
        border: none;
        border-radius: 4px;
        width: 100%;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-main">Retention Strategy Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Agentic AI system for churn intervention planning</div>', unsafe_allow_html=True)

@st.cache_resource
def load_ml_model():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "trained_models.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

ml_model = load_ml_model()

col_left, col_right = st.columns([4, 6])

with col_left:
    st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)
    customer_id = st.text_input("Customer ID", value="CUST-1001")
    
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        age = st.number_input("Age", 18, 100, 45)
        balance = st.number_input("Balance", 0.0, 300000.0, 100000.0)
        has_crcard = st.selectbox("Has Credit Card", [1, 0])
        credit_score = st.number_input("Credit Score", 300, 850, 650)
    with col_l2:
        tenure = st.number_input("Tenure", 0, 10, 5)
        num_products = st.selectbox("Num Of Products", [1, 2, 3, 4])
        is_active = st.selectbox("Is Active Member", [1, 0])
        salary = st.number_input("Estimated Salary", 0.0, 300000.0, 80000.0)
        
    run_agent = st.button("Run Agent")

report = None

with col_right:
    st.markdown('<div class="section-header">Agent execution status</div>', unsafe_allow_html=True)
    status_containers = {
        "risk_profiler_node": st.empty(),
        "strategy_retriever_node": st.empty(),
        "intervention_planner_node": st.empty(),
        "report_generator_node": st.empty()
    }
    
    for k, v in status_containers.items():
        v.markdown(f"{k} | Pending")
        
    if run_agent:
        if not ml_model:
            st.markdown('<div style="color:#e63946;font-weight:bold;">ML model not found. Cannot compute churn probability.</div>', unsafe_allow_html=True)
        else:
            inputs = {
                "CustomerID": customer_id,
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_crcard,
                "IsActiveMember": is_active,
                "EstimatedSalary": salary
            }
            
            df = pd.DataFrame([inputs])
            prob = ml_model.predict_proba(df)[0][1]
            
            agent = create_agent_graph()
            initial_state = {
                "customer_features": inputs,
                "churn_probability": float(prob),
                "risk_tier": "",
                "risk_summary": "",
                "retrieved_strategies": [],
                "intervention_reasoning": "",
                "final_report": {},
                "error": None
            }
            
            current_state = initial_state
            for output in agent.stream(initial_state):
                for node_name, state_update in output.items():
                    current_state.update(state_update)
                    status_containers[node_name].markdown(f"**{node_name}** | Complete")
                    
            if current_state.get("error"):
                st.markdown(f'<div style="color:#e63946;font-weight:bold;">Agent failed: {current_state["error"]}</div>', unsafe_allow_html=True)
                report = current_state.get("final_report", {})
            else:
                report = current_state.get("final_report", {})

if report:
    st.markdown("---")
    
    # Card 1 - Risk Assessment
    st.markdown('<div class="flat-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="margin-top:0;">Risk Assessment</div>', unsafe_allow_html=True)
    tier = report.get("risk_tier", "Low")
    badge_class = "badge-low"
    if tier in ["High", "Critical"]: badge_class = "badge-high"
    elif tier == "Medium": badge_class = "badge-medium"
    
    st.markdown(f"**Customer ID:** {report.get('customer_id', '')}")
    p = report.get('churn_probability', 0)
    st.markdown(f"**Churn Probability:** {p*100:.1f}%")
    st.markdown(f"**Risk Tier:** <span class='{badge_class}'>{tier}</span>", unsafe_allow_html=True)
    st.markdown(f"**Risk Summary:** {', '.join(report.get('key_risk_factors', []))}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Card 2 - Recommended Actions
    st.markdown('<div class="flat-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="margin-top:0;">Intervention Plan</div>', unsafe_allow_html=True)
    
    actions = report.get("recommended_actions", [])
    if actions:
        table_html = "<table style='width:100%; border-collapse:collapse;'>"
        table_html += "<tr style='border-bottom:1px solid #ddd; text-align:left;'><th>Priority</th><th>Action</th><th>Rationale</th></tr>"
        for act in actions:
            pri = str(act.get("priority", "Low"))
            color = "#2a9d8f"
            if pri == "High": color = "#e63946"
            elif pri == "Medium": color = "#e9c46a"
            
            table_html += f"<tr style='border-bottom:1px solid #eee;'>"
            table_html += f"<td style='color:{color}; font-weight:bold;'>{pri}</td>"
            table_html += f"<td>{act.get('action', '')}</td>"
            table_html += f"<td>{act.get('rationale', '')}</td>"
            table_html += "</tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.write("No actions recommended.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Card 3 - Retention Offer & Impact
    st.markdown('<div class="flat-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="margin-top:0;">Proposed Offer</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Retention Offer:**")
        st.markdown(report.get("retention_offer", "N/A"))
    with c2:
        st.markdown("**Expected Impact:**")
        st.markdown(report.get("expected_impact", "N/A"))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Card 4 - Sources & Disclaimer
    st.markdown('<div class="flat-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="margin-top:0;">References & Disclosure</div>', unsafe_allow_html=True)
    st.markdown("**Sources:**")
    for s in report.get("sources", []):
        st.markdown(f"- {s}")
    st.markdown(f"<br><span style='color:gray; font-style:italic;'>{report.get('ethical_disclaimer', '')}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    json_str = json.dumps(report, indent=2)
    st.download_button("Download Report (JSON)", data=json_str, file_name="retention_report.json", mime="application/json")


st.markdown("---")
st.markdown('<div class="section-header">Agent Workflow Diagram</div>', unsafe_allow_html=True)
st.graphviz_chart('''
digraph G {
    rankdir=LR;
    node [shape=box, style=filled, fillcolor="#f4f4f4", fontcolor="#1a1a2e", color="#dddddd", fontname="system-ui"];
    edge [color="#1a1a2e"];
    
    START [shape=circle, width=0.5, style=filled, fillcolor="#2a9d8f", fontcolor="white"];
    END [shape=circle, width=0.5, style=filled, fillcolor="#e63946", fontcolor="white"];
    
    risk_profiler [label="Risk\nProfiler"];
    strategy_retriever [label="Strategy\nRetriever"];
    intervention_planner [label="Intervention\nPlanner"];
    report_generator [label="Report\nGenerator"];
    error_handler [label="Error\nHandler", fillcolor="#e63946", fontcolor="white"];
    
    START -> risk_profiler;
    risk_profiler -> strategy_retriever;
    strategy_retriever -> intervention_planner;
    intervention_planner -> report_generator;
    report_generator -> END;
    
    risk_profiler -> error_handler [color="#e63946", style="dashed"];
    strategy_retriever -> error_handler [color="#e63946", style="dashed"];
    intervention_planner -> error_handler [color="#e63946", style="dashed"];
    report_generator -> error_handler [color="#e63946", style="dashed"];
}
''')
