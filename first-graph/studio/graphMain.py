# from langchain_core.messages import AIMessage
# from langchain_core.runnables import RunnableLambda
# from langgraph.graph import START, StateGraph, MessagesState
# from langgraph.prebuilt import tools_condition, ToolNode
# from langchain_core.tools import Tool
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.schema.runnable import RunnableLambda
# from operator import itemgetter
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from dotenv import load_dotenv
# from typing_extensions import TypedDict
# import os

# load_dotenv()


# # Define StateSchema schema
# class StateSchema(TypedDict):
#     messages: list

# # Define the tools to be used
# def answer_like_marc(input: str) -> str:
#     """
#     This function takes a string input and returns a string output.
#     It is designed to mimic how Marc speaks
#     """
#     return "Yooooo " + input

# # Define an LLM answering node
# def llm_answer_node(state):
#     print(f"State[messages]: {state['messages']}")
#     messages = state["messages"]
#     question = messages[-1]  # assuming last message is the question
#     docs = retriever.get_relevant_documents(question)

#     def format_docs(docs):
#         return "\n\n".join([doc.page_content for doc in docs])
    
#     def return_question(state):
#         return state["messages"][-1]
    
#     print(f"Docs: {format_docs(docs)}")
#     # Define chain
#     llm = ChatOpenAI(model_name="gpt-4o")

#     # Wrap the function into a proper tool
#     answer_like_marc_tool = Tool(
#         name="answer_like_marc",
#         description="Mimics how Marc speaks by adding 'Yooooo' at the beginning.",
#         func=answer_like_marc
#     )

#     llm_with_tools = llm.bind_tools([answer_like_marc_tool])

#     prompt = PromptTemplate.from_template("""
#     Use the following context to answer the question. Answer like Marc would.

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """)

#     rag_chain = (
#     {
#         "context": lambda x: format_docs(docs),
#         "question": RunnableLambda(return_question)
#     }
#     | prompt
#     | llm_with_tools
#     | StrOutputParser()
# )

#     answer = rag_chain.invoke(state)

#     print(f"Answer: {answer}")
#     return {"messages": messages + [AIMessage(content=answer)]}


# tools = [answer_like_marc]

# # Load the document from the specified file path
# file_path = os.path.join(os.path.dirname(__file__), "../documents/sample.txt")
# loader = TextLoader(file_path)
# documents = loader.load()

# # Split the document into smaller chunks for better processing
# # Each chunk has a maximum size of 500 characters with a 50-character overlap
# splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# texts = splitter.split_documents(documents)

# # Generate vector embeddings for the text chunks
# embeddings = OpenAIEmbeddings()
# # Create a FAISS vector store to index the embeddings for efficient retrieval
# vectorstore = FAISS.from_documents(texts, embeddings)

# # Convert the vector store into a retriever object for fetching relevant chunks
# retriever = vectorstore.as_retriever()

# # Create a graph builder with the state schema
# builder = StateGraph(StateSchema)

# # Add LangGraph nodes (each is a function wrapped in RunnableLambda)
# builder.add_node("answerNode", RunnableLambda(llm_answer_node))
# builder.add_node("tools", ToolNode(tools))

# builder.add_edge(START, "answerNode")
# builder.add_conditional_edges("answerNode", tools_condition)
# builder.add_edge("tools", "answerNode")

# # Compile the graph into a runnable object
# graph = builder.compile()