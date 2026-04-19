import os
import json
import datetime
from .state import AgentState
from ..rag.retriever import VectorStoreRetriever

# Import LLM integrations
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

def get_llm_response(prompt: str, temperature: float = 0.5, require_json: bool = False) -> str:
    """Helper macro to call whichever API is configured."""
    system_instruction = "Base all recommendations strictly on the retrieved context. If the context does not support a recommendation, state that explicitly. Do not fabricate statistics or guarantees."
    
    if "GEMINI_API_KEY" in os.environ and genai:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        # Use gemini-1.5-flash as specified or gemini-2.0-flash depending on SDK but 1.5 is safe
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_instruction
        )
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json" if require_json else "text/plain"
        )
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
        
    elif "GROQ_API_KEY" in os.environ and Groq:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        if require_json:
            messages[0]["content"] += " YOU MUST RESPOND ONLY WITH VALID JSON."
            
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    else:
        raise ValueError("Neither GEMINI_API_KEY nor GROQ_API_KEY is configured correctly.")

def extract_json(response: str) -> dict:
    try:
        # try parse directly
        return json.loads(response)
    except json.JSONDecodeError:
        # Find json block
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            return json.loads(response[start:end+1])
        raise ValueError("Could not parse JSON from response")

def risk_profiler_node(state: AgentState) -> dict:
    churn_prob = state.get("churn_probability", 0.0)
    features = state.get("customer_features", {})
    
    try:
        if churn_prob >= 0.8:
            tier = "Critical"
        elif churn_prob >= 0.5:
            tier = "High"
        elif churn_prob >= 0.2:
            tier = "Medium"
        else:
            tier = "Low"
            
        summary = f"Customer has a {churn_prob:.1%} probability of churning. They fall into the {tier} risk tier based on their profile. "
        summary += f"Key factors identified relate to tenure {features.get('Tenure', 'N/A')} and balance {features.get('Balance', 'N/A')}."
        return {"risk_tier": tier, "risk_summary": summary}
    except Exception as e:
        return {"error": f"Risk Profiler Error: {str(e)}"}

def strategy_retriever_node(state: AgentState) -> dict:
    try:
        retriever = VectorStoreRetriever()
        risk_summary = state.get("risk_summary", "")
        # Query ChromaDB with the risk profile
        chunks = retriever.retrieve_strategies(query=risk_summary, top_k=5)
        return {"retrieved_strategies": chunks}
    except Exception as e:
        return {"error": f"Strategy Retriever Error: {str(e)}"}

def intervention_planner_node(state: AgentState) -> dict:
    if state.get("error"): return {}
    try:
        summary = state.get("risk_summary", "")
        strategies = state.get("retrieved_strategies", [])
        
        context_block = "\n".join([f"- {s}" for s in strategies])
        prompt = f"""
        Given the following customer risk summary:
        {summary}
        
        And the following retrieved strategies:
        [CONTEXT]
        {context_block}
        [/CONTEXT]
        
        Reason through which strategies apply and why. Provide a chain-of-thought analysis explaining the intervention plan.
        """
        reasoning = get_llm_response(prompt, temperature=0.5, require_json=False)
        return {"intervention_reasoning": reasoning}
    except Exception as e:
        return {"error": f"Intervention Planner Error: {str(e)}"}

def report_generator_node(state: AgentState) -> dict:
    if state.get("error"): return __fallback_report(state)
    try:
        features = state.get("customer_features", {})
        customer_id = str(features.get("CustomerID", "UNKNOWN"))
        tier = state.get("risk_tier", "Unknown")
        prob = state.get("churn_probability", 0.0)
        reasoning = state.get("intervention_reasoning", "")
        strategies = state.get("retrieved_strategies", [])
        
        context_block = "\n".join([f"- {s}" for s in strategies])
        prompt = f"""
        Based on the intervention reasoning:
        {reasoning}
        
        And context:
        [CONTEXT]
        {context_block}
        [/CONTEXT]
        
        Produce a structured JSON retention report with EXACTLY these fields:
        {{
          "customer_id": "{customer_id}",
          "risk_tier": "{tier}",
          "churn_probability": {prob},
          "key_risk_factors": ["string"],
          "recommended_actions": [
            {{"action": "string", "rationale": "string", "priority": "High|Medium|Low"}}
          ],
          "retention_offer": "string",
          "expected_impact": "string",
          "sources": ["string"],
          "ethical_disclaimer": "string",
          "generated_at": "{datetime.datetime.now().isoformat()}"
        }}
        
        Respond ONLY with the JSON object.
        """
        resp_text = get_llm_response(prompt, temperature=0.2, require_json=True)
        report_json = extract_json(resp_text)
        return {"final_report": report_json}
    except Exception as e:
        return {"error": f"Report Generator Error: {str(e)}", "final_report": __fallback_report(state)["final_report"]}

def __fallback_report(state: AgentState) -> dict:
    features = state.get("customer_features", {})
    return {
        "final_report": {
           "customer_id": str(features.get("CustomerID", "UNKNOWN")),
           "risk_tier": state.get("risk_tier", "Unknown"),
           "churn_probability": state.get("churn_probability", 0.0),
           "key_risk_factors": ["Error generating insights"],
           "recommended_actions": [{"action": "Manual review required", "rationale": "System error", "priority": "High"}],
           "retention_offer": "None",
           "expected_impact": "Unknown",
           "sources": [],
           "ethical_disclaimer": "This is a fallback report. An error occurred during generation.",
           "generated_at": datetime.datetime.now().isoformat()
        }
    }
