# Simple Streamlit UI
# Just a text input, a submit button, and the response shown below
import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.title("Agentic RAG - Financial Market Intelligence")
st.write("Ask questions about Tesla 10-K filings, financial news and analyst reports.")

# Keep chat history in session
if "history" not in st.session_state:
    st.session_state.history = []

use_agents = st.checkbox("Use multi-agent workflow (LangGraph)", value=True)
query = st.text_input("Enter your question:")

if st.button("Submit") and query.strip():
    chat_history_str = "\n".join(
        [f"User: {q}\nBot: {a}" for q, a in st.session_state.history]
    )
    try:
        with st.spinner("Thinking..."):
            res = requests.post(
                API_URL,
                json={
                    "query": query,
                    "chat_history": chat_history_str,
                    "use_agents": use_agents,
                },
                timeout=180,
            )
            data = res.json()

        response_text = data.get("response", "")
        st.subheader("Response")
        st.write(response_text)

        # Show extra agent outputs if available
        if use_agents:
            if data.get("analysis"):
                st.subheader("Analysis")
                st.write(data["analysis"])
            if data.get("portfolio_advice"):
                st.subheader("Portfolio Advice")
                st.write(data["portfolio_advice"])
            if data.get("risk_assessment"):
                st.subheader("Risk Assessment")
                st.write(data["risk_assessment"])

        # Retrieved documents
        sources = data.get("sources", [])
        if sources:
            st.subheader("Retrieved Context")
            for i, s in enumerate(sources, 1):
                with st.expander(f"Document {i}"):
                    st.write(s)

        # Save to history
        st.session_state.history.append((query, response_text))

    except Exception as e:
        st.error(f"Error calling API: {e}")

# Show chat history
if st.session_state.history:
    st.subheader("Chat History")
    for q, a in st.session_state.history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {a}")
        st.write("---")
