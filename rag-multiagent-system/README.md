# Agentic RAG - Financial Market Intelligence (Tesla)

A simple Agentic RAG capstone project built with **FastAPI**, **Streamlit**, **LangChain**, **LangGraph**, **FAISS**, and **HuggingFace embeddings**.

It simulates a **Financial Market Intelligence** assistant focused on **Tesla (2024-2025)** using three datasets:

1. Tesla 10-K filings
2. Tesla financial news
3. Tesla market analysis / analyst reports

---

## Project Structure

```
rag-multiagent-system/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── rag/
│   │   ├── rag_pipeline.py        # Main RAG workflow
│   │   ├── chunking.py            # Metadata-aware chunking for all 3 datasets
│   │   ├── retriever.py
│   │   └── hybrid_search.py
│   ├── agents/
│   │   └── agents.py              # All agents + LangGraph workflow
│   └── evaluation/
│       ├── rouge_eval.py
│       └── ragas_eval.py
├── vectorstores/
│   ├── finance_10k/
│   ├── financial_news/
│   └── market_analysis/
├── data/
│   ├── finance_10k/               # Drop Tesla 10-K PDFs here
│   ├── financial_news/            # Drop news PDFs here
│   └── market_analysis/           # Drop analyst report PDFs here
├── streamlit_app/
│   └── main.py
├── .github/workflows/ci-cd.yml
├── Dockerfile
├── requirements.txt
├── .env
└── README.md
```

---

## Setup

1. Clone the repo and `cd` into the folder.
2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate     # or  venv\Scripts\activate  on Windows
pip install -r requirements.txt
```

3. Add your `GOOGLE_API_KEY` to the `.env` file (free key from Google AI Studio).

4. Place your Tesla PDFs in the `data/` folders:
   - `data/finance_10k/` -> Tesla 10-K PDFs
   - `data/financial_news/` -> news article PDFs
   - `data/market_analysis/` -> analyst report PDFs

5. Build the vectorstores (run once):

```bash
python -m app.rag.rag_pipeline
```

---

## Run

### Option 1 - Locally

```bash
# Terminal 1 - FastAPI backend
uvicorn app.main:app --reload

# Terminal 2 - Streamlit UI
streamlit run streamlit_app/main.py
```

### Option 2 - Docker

```bash
docker build -t rag-multiagent-system .
docker run -p 8000:8000 --env-file .env rag-multiagent-system
```

Then start Streamlit locally:

```bash
streamlit run streamlit_app/main.py
```

---

## API

### POST `/query`

```json
{
  "query": "Should I invest in Tesla in 2025?",
  "chat_history": "",
  "use_agents": true
}
```

Response includes the final response, analysis, portfolio advice, risk assessment, retrieved sources, and validation message.

---

## Multi-Agent Workflow (LangGraph)

```
START -> Planner -> Retriever -> Analysis -> Portfolio -> Risk -> Final -> END
```

- **Planner Agent** - decides what info is needed
- **Retriever Agent** - calls the RAG tool, validates retrieval
- **Analysis Agent** - generates market analysis
- **Portfolio Agent** - suggests allocation
- **Risk Agent** - lists key risks
- **Final** - composes structured response

---

## Evaluation

- `app/evaluation/rouge_eval.py` -> ROUGE-1 / ROUGE-2 / ROUGE-L
- `app/evaluation/ragas_eval.py` -> Faithfulness, Answer Relevancy, Context Precision

---

## Why separate vectorstores?

- Each dataset has different structure and metadata
- Different chunking strategies
- Easier debugging and filtering
- Cleaner retrieval per source
