from fastapi import FastAPI, Query  # Import FastAPI for building the web API
from pydantic import BaseModel  # Import BaseModel for request validation
from app.retrieval_agent import build_agent  # Import the function to build the retrieval agent
from dotenv import load_dotenv  # Import dotenv to load environment variables from a .env file

# Load environment variables from a .env file
load_dotenv()

# Create a FastAPI application instance
app = FastAPI()

# Build the retrieval-based question-answering agent
qa_chain = build_agent()

# Define a Pydantic model for validating the incoming request body
class QueryRequest(BaseModel):
    question: str  # The request must include a "question" field of type string

# Define a POST endpoint at "/ask" to handle user queries
@app.post("/ask")
async def ask(request: QueryRequest):
    # Use the QA chain to process the user's question and generate an answer
    response = qa_chain.run(request.question)
    # Return the answer as a JSON response
    return {"answer": response}