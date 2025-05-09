from fastapi import FastAPI  # Import FastAPI for building the web API
from pydantic import BaseModel  # Import BaseModel for request validation
from app.retrieval_agent import build_agent  # Import the function to build the retrieval agent
from dotenv import load_dotenv  # Import dotenv to load environment variables from a .env file
from app.langgraph_graph import build_rag_graph  # Import the function to build the LangGraph graph

# Load environment variables from a .env file
load_dotenv()

# Ask the user to choose between the simple LangChain version or the LangGraph version
print("Choose the mode to run:")
print("1: Simple LangChain version")
print("2: LangGraph version")
choice = input("Enter your choice (1 or 2): ").strip()

# Create a FastAPI application instance
app = FastAPI()

if choice == "1":
    # Build the retrieval-based question-answering agent (Simple LangChain version)
    qa_chain = build_agent()

    # Define a Pydantic model for validating the incoming request body
    class QueryRequest(BaseModel):
        question: str  # The request must include a "question" field of type string

    # Define a POST endpoint at "/ask" to handle user queries
    @app.post("/ask")
    async def ask(request: QueryRequest):
        # Use the QA chain to process the user's question and generate an answer
        response = qa_chain.run(request.question)
        print("Responding Using Simple LangChain Implementation")
        # Return the answer as a JSON response
        return {"answer": response}

elif choice == "2":
    # Build the LangGraph graph (LangGraph version)
    rag_graph = build_rag_graph()

    # Define a Pydantic model for validating the incoming request body
    class QueryRequest(BaseModel):
        question: str  # The request must include a "question" field of type string

    # Define a POST endpoint at "/ask" to handle user queries
    @app.post("/ask")
    async def ask(request: QueryRequest):
        # Use the LangGraph graph to process the user's question and generate an answer
        response = rag_graph.invoke({"question": request.question})
        print("Responding Using LangGraph Implementation")
        # Return the answer as a JSON response
        return {"answer": response}

else:
    # Exit the program if the user enters an invalid choice
    print("Invalid choice. Exiting...")
    exit()