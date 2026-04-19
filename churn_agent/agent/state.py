from typing import TypedDict, List, Dict, Optional, Any

class AgentState(TypedDict):
  customer_features: dict
  churn_probability: float
  risk_tier: str
  risk_summary: str
  retrieved_strategies: List[str]
  intervention_reasoning: str
  final_report: Dict[str, Any]
  error: Optional[str]
