from langgraph.graph import StateGraph
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableLambda
from app.retrieval_agent import build_agent

# Load your existing QA chain
qa_chain = build_agent()

# Define a retrieval node
def retrieve_node(state):
    question = state["question"]
    docs = qa_chain.retriever.get_relevant_documents(question)
    return {"question": question, "docs": docs}

# Define an LLM answering node
def llm_answer_node(state):
    context = "\n\n".join(doc.page_content for doc in state["docs"])
    full_prompt = f"Answer this question using the context below:\n\n{context}\n\nQuestion: {state['question']}"
    answer = qa_chain.llm.invoke(full_prompt)
    return {"question": state["question"], "docs": state["docs"], "answer": answer.content}