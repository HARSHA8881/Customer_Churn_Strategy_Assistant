from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    risk_profiler_node,
    strategy_retriever_node,
    intervention_planner_node,
    report_generator_node
)

def create_agent_graph():
    # Initialize StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("risk_profiler_node", risk_profiler_node)
    workflow.add_node("strategy_retriever_node", strategy_retriever_node)
    workflow.add_node("intervention_planner_node", intervention_planner_node)
    workflow.add_node("report_generator_node", report_generator_node)
    
    # Define linear execution flow
    workflow.set_entry_point("risk_profiler_node")
    workflow.add_edge("risk_profiler_node", "strategy_retriever_node")
    workflow.add_edge("strategy_retriever_node", "intervention_planner_node")
    workflow.add_edge("intervention_planner_node", "report_generator_node")
    workflow.add_edge("report_generator_node", END)
    
    # Compile graph
    app = workflow.compile()
    return app
