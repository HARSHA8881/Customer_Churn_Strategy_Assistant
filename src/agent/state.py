from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    customer_data: Dict[str, Any]
    churn_score: float
    risk_level: str
    churn_drivers: List[str]
    retrieved_strategies: List[str]
    final_recommendations: Dict[str, Any]
    error: str
