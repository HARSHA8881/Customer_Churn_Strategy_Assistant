# AI Customer Retention Strategy Assistant

This project is a complete, production-ready GenAI + Agentic AI system that predicts customer churn and generates hyper-personalized retention strategies using LangGraph, RAG (ChromaDB), and Streamlit.

## Features
- **Machine Learning Integration**: Uses Scikit-Learn (Logistic Regression) to predict baseline churn probability.
- **Agentic Workflow (LangGraph)**:
  - **Risk Analyzer**: Analyzes key churn drivers using an LLM.
  - **Retriever**: Queries a local vector database (ChromaDB) to fetch context-specific retention strategies.
  - **Strategy Planner**: Synthesizes profile data and retrieved strategies into actionable steps.
- **Premium UI**: Built with Streamlit, applying custom CSS for a modern, sleek appearance.
- **Strict Structured Outputs**: Ensures the LLM outputs reasoning and recommendations in a strict JSON format.

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- A Groq API Key (`GROQ_API_KEY`)

### 2. Installation
Clone the repository, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your API key:
```env
GROQ_API_KEY="your-groq-api-key-here"
```

### 4. Running the Application
To run the Streamlit app locally:
```bash
streamlit run app.py
```
*Note: On the first run, the app will automatically generate synthetic data, train the baseline ML model, and compile the ChromaDB vector store.*

## Deployment Steps (Streamlit Cloud)
1. Push this code to a public/private GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app", link your repository, and select `app.py` as the entrypoint.
4. Under "Advanced Settings" during deployment, add your `GROQ_API_KEY` to the Secrets section.
5. Click **Deploy**.

## Testing Notes
- If an agent node encounters a parsing error from the LLM, the system catches it gracefully and presents the partial logic in the UI without crashing.
# Customer_Churn_Strategy_Assistant
