from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def build_agent():
    # Load the document from the specified file path
    loader = TextLoader("app/documents/sample.txt")
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
    # Initialize the language model with the specified model and temperature
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Create a RetrievalQA chain that uses the retriever and language model
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    # Return the QA chain for answering queries
    return qa_chain