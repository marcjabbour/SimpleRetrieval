# SimpleRetrieval



## Retrieval-Agent:

Leveraging LangChain functions to create a simple RAG implementation. Deploying the app using a FastAPI applicatoin instance and invoking the exposed function via REST API.

- In one terminal, run server with: `uvicorn app.main:app --reload`

- In separate terminal, interact with endpoint with:
```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is LangChain?"}'
```

## First-Graph:

Leveraging LangGraph to create an agentic workflow. I've bound a tool to the LLM which will optionally check the `/documents` folder to see if there's information relevant to the question that was asked. If so, that relevant info will substantiate the response from the LLM. The relevant file is `/graphAlt.py`.

- In `/studio` , run `langgraph dev` to spin up a local instance of Langgraph Studio and interact with the graph.