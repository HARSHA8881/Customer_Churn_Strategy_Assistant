import streamlit as st
import os
import subprocess

st.set_page_config(
    page_title="AI Customer Retention Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Customer Churn Prediction & Agentic Retention Strategy")

st.markdown("""
Welcome to the Customer Retention Assistant (Milestone 2).

### Available Pages:
**1. Upload and Predict:** (Milestone 1) Classical machine learning pipelines to predict churn risk.
**2. Retention Agent:** (Milestone 2) LangGraph & RAG-based agentic AI application to generate structured retention strategies.

Please navigate using the sidebar.
""")

# Initialize ChromaDB KB if missing
if not os.path.exists("./chroma_db"):
    st.markdown("<p style='color: #60A5FA;'>Initializing Knowledge Base for the first time...</p>", unsafe_allow_html=True)
    try:
        subprocess.run(["python", "rag/build_kb.py"])
        st.markdown("<p style='color: #60A5FA; font-weight: bold;'>Knowledge Base successfully built.</p>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<p style='color: #1D4ED8;'>Failed to build Knowledge Base: {e}</p>", unsafe_allow_html=True)
