from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
import os


# Define the tools to be used
def get_supporting_docs(query: str) -> str:
    """
    Retrieve the top 3 relevant documents for the given query.
    """
    retriever = generate_docs_retriever()
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs[:3]])

def generate_docs_retriever():
    # Load the document from the specified file path
    file_path = os.path.join(os.path.dirname(__file__), "../documents/sample.txt")
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into smaller chunks for better processing
    # Each chunk has a maximum size of 500 characters with a 50-character overlap
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    # Generate vector embeddings for the text chunks
    embeddings = OpenAIEmbeddings()
    # Create a FAISS vector store to index the embeddings for efficient retrieval
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Convert the vector store into a retriever object for fetching relevant chunks
    retriever = vectorstore.as_retriever()
    return retriever


tools = [get_supporting_docs]

# Define LLM with bound tools
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant which leverage supporting documentation if relevant.")

# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()
