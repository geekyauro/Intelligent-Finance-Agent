# FastAPI - REST API entry point
from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.rag_pipeline import run_pipeline
from app.agents.agents import run_agentic_workflow

app = FastAPI()


class Query(BaseModel):
    query: str
    chat_history: str = ""
    use_agents: bool = True


@app.get("/")
def home():
    return {"message": "Agentic RAG API Running"}


@app.post("/query")
def ask(q: Query):
    # If use_agents=True -> run the full LangGraph multi-agent workflow
    # else just run the plain RAG pipeline
    if q.use_agents:
        result = run_agentic_workflow(q.query, q.chat_history)
        return {
            "response": result.get("final_response", ""),
            "analysis": result.get("analysis", ""),
            "portfolio_advice": result.get("portfolio_advice", ""),
            "risk_assessment": result.get("risk_assessment", ""),
            "sources": result.get("retrieved_docs", []),
            "validation": result.get("validation", ""),
        }
    else:
        result = run_pipeline(q.query, q.chat_history)
        return {
            "response": result.get("response", ""),
            "sources": result.get("context", []),
            "validation": result.get("validation", ""),
        }
