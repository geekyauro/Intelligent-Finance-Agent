# All agents in a single file - implemented as simple functions
# LangGraph is used to orchestrate them in a workflow

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

from app.rag.rag_pipeline import (
    retrieve_from_all_sources,
    generate_response,
    validate_retrieval,
    get_llm,
    rag_tool,
)


# ----------------------------
# Shared state passed across agents (conversational memory lives here)
# ----------------------------
class AgentState(TypedDict):
    question: str
    plan: str
    retrieved_docs: List[str]
    analysis: str
    portfolio_advice: str
    risk_assessment: str
    final_response: str
    chat_history: str
    validation: str


# ----------------------------
# Planner Agent
# ----------------------------
def planner_agent(state: AgentState) -> AgentState:
    # Decides what needs to be done based on the question
    llm = get_llm()
    prompt = f"""You are a financial planner agent. Given the user question,
write a short 2-3 line plan describing what information is needed to answer it.

Question: {state['question']}

Plan:"""
    result = llm.invoke(prompt)
    plan = result.content if hasattr(result, "content") else str(result)
    state["plan"] = plan
    print(f"[Planner] Plan generated.")
    return state


# ----------------------------
# Retriever Agent (calls the RAG tool)
# ----------------------------
def retriever_agent(state: AgentState) -> AgentState:
    # Retrieves documents from all three vectorstores using RAG as a tool
    docs = retrieve_from_all_sources(state["question"], k=3)
    state["retrieved_docs"] = [d.page_content for d in docs]

    # Retrieval validation
    is_valid, msg = validate_retrieval(state["question"], docs)
    state["validation"] = msg
    print(f"[Retriever] Retrieved {len(docs)} docs. Validation: {msg}")

    # Generate a context-grounded answer here too
    if is_valid:
        state["analysis"] = generate_response(
            state["question"], docs, state.get("chat_history", "")
        )
    else:
        state["analysis"] = "Insufficient context for analysis."
    return state


# ----------------------------
# Analysis Agent
# ----------------------------
def analysis_agent(state: AgentState) -> AgentState:
    # Analyzes the retrieved info and produces market insights
    llm = get_llm()
    context = "\n".join(state.get("retrieved_docs", []))[:3000]
    prompt = f"""You are a financial Analysis Agent.
Based on the retrieved context below, produce a short market analysis (3-5 sentences)
for the user question. Stay grounded in the context.

Context:
{context}

Question: {state['question']}

Analysis:"""
    result = llm.invoke(prompt)
    state["analysis"] = result.content if hasattr(result, "content") else str(result)
    print(f"[Analysis] Done.")
    return state


# ----------------------------
# Portfolio Agent
# ----------------------------
def portfolio_agent(state: AgentState) -> AgentState:
    # Suggests portfolio allocation ideas based on the analysis
    llm = get_llm()
    prompt = f"""You are a Portfolio Agent.
Based on the analysis below, suggest a simple portfolio allocation idea
(aggressive / moderate / conservative) and a short justification.

Analysis:
{state.get('analysis', '')}

Portfolio Suggestion:"""
    result = llm.invoke(prompt)
    state["portfolio_advice"] = (
        result.content if hasattr(result, "content") else str(result)
    )
    print(f"[Portfolio] Done.")
    return state


# ----------------------------
# Risk Assessment Agent
# ----------------------------
def risk_agent(state: AgentState) -> AgentState:
    # Evaluates risks from the retrieved context and analysis
    llm = get_llm()
    context = "\n".join(state.get("retrieved_docs", []))[:2000]
    prompt = f"""You are a Risk Assessment Agent.
List 2-3 key risks for the user's question, based ONLY on the context and analysis.

Context:
{context}

Analysis:
{state.get('analysis', '')}

Risks:"""
    result = llm.invoke(prompt)
    state["risk_assessment"] = (
        result.content if hasattr(result, "content") else str(result)
    )
    print(f"[Risk] Done.")
    return state


# ----------------------------
# Validation / Final Response Step
# ----------------------------
def final_response_agent(state: AgentState) -> AgentState:
    # Combines everything into a structured final answer
    final = f"""## Market Analysis
{state.get('analysis', 'N/A')}

## Portfolio Suggestion
{state.get('portfolio_advice', 'N/A')}

## Risk Assessment
{state.get('risk_assessment', 'N/A')}

## Retrieval Validation
{state.get('validation', 'N/A')}
"""
    state["final_response"] = final
    print(f"[Final] Response composed.")
    return state


# ----------------------------
# LangGraph workflow
# ----------------------------
def build_agent_graph():
    # START -> Planner -> Retriever -> Analysis -> Portfolio -> Risk -> Final -> END
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("portfolio", portfolio_agent)
    graph.add_node("risk", risk_agent)
    graph.add_node("final", final_response_agent)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "analysis")
    graph.add_edge("analysis", "portfolio")
    graph.add_edge("portfolio", "risk")
    graph.add_edge("risk", "final")
    graph.add_edge("final", END)

    return graph.compile()


# Single compiled graph instance (built once and reused)
AGENT_GRAPH = None


def get_agent_graph():
    global AGENT_GRAPH
    if AGENT_GRAPH is None:
        AGENT_GRAPH = build_agent_graph()
    return AGENT_GRAPH


def run_agentic_workflow(question: str, chat_history: str = ""):
    # Entry point - runs the full agent workflow for a given question
    graph = get_agent_graph()
    initial_state = {
        "question": question,
        "plan": "",
        "retrieved_docs": [],
        "analysis": "",
        "portfolio_advice": "",
        "risk_assessment": "",
        "final_response": "",
        "chat_history": chat_history,
        "validation": "",
    }
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # Quick test
    result = run_agentic_workflow("Should I invest in Tesla in 2025?")
    print("\n===== FINAL RESPONSE =====")
    print(result["final_response"])
