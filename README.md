# SimpleRetrieval


Run server with: `uvicorn app.main:app --reload`

In separate terminal, interact with endpoint with:
```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is LangChain?"}'
```