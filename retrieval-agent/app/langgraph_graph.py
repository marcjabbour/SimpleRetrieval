# langgraph_graph.py

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from app.graph_nodes import retrieve_node, llm_answer_node

def build_rag_graph():
    # Define the state schema or input/output for the graph
    state_schema = {
        "question": str,  # Define the input type
        "answer": str     # Define the output type
    }

    # Create a graph builder with the state schema
    builder = StateGraph(state_schema=state_schema)

    # Add LangGraph nodes (each is a function wrapped in RunnableLambda)
    builder.add_node("retrieve", RunnableLambda(retrieve_node))
    builder.add_node("answer", RunnableLambda(llm_answer_node))

    # Define graph flow: start → retrieve → answer → end
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "answer")
    builder.set_finish_point("answer")

    # Compile the graph into a runnable object
    return builder.compile()