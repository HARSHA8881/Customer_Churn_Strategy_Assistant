from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    risk_analyzer_node,
    retriever_node,
    strategy_planner_node,
    response_generator_node
)

def create_agent_graph():
    # Initialize StateGraph with our typing
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("RiskAnalyzer", risk_analyzer_node)
    workflow.add_node("Retriever", retriever_node)
    workflow.add_node("StrategyPlanner", strategy_planner_node)
    workflow.add_node("ResponseGenerator", response_generator_node)
    
    # Define edges
    # We will run them linearly for this strategy assistant
    workflow.set_entry_point("RiskAnalyzer")
    workflow.add_edge("RiskAnalyzer", "Retriever")
    workflow.add_edge("Retriever", "StrategyPlanner")
    workflow.add_edge("StrategyPlanner", "ResponseGenerator")
    workflow.add_edge("ResponseGenerator", END)
    
    # Compile graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    # Small test
    graph = create_agent_graph()
    print("Graph compiled successfully!")
