# RAG Pipeline - main file containing all the major RAG functions
# Keeps things simple: loading, chunking, embeddings, vectorstore, retrieval, generation, validation
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from app.rag.chunking import (
    chunk_10k_filings,
    chunk_financial_news,
    chunk_market_analysis,
)

load_dotenv()


# ----------------------------
# Embeddings
# ----------------------------
def get_embeddings():
    # Free HuggingFace embedding model - small and fast
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ----------------------------
# LLM
# ----------------------------
def get_llm():
    # Using Gemini (free tier). Can swap with ChatOpenAI if needed.
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        max_output_tokens=2000,
        temperature=0.2,
    )


# ----------------------------
# Data Loading (PDFs only - using PyPDFLoader)
# ----------------------------
def load_pdfs_from_folder(folder_path):
    # Loads all PDF files inside a folder and returns a list of documents
    docs = []
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return docs

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(file_path)
            file_docs = loader.load()
            docs.extend(file_docs)
            print(f"Loaded: {file_name} ({len(file_docs)} pages)")
    return docs


def load_finance_10k():
    return load_pdfs_from_folder("data/finance_10k")


def load_financial_news():
    return load_pdfs_from_folder("data/financial_news")


def load_market_analysis():
    return load_pdfs_from_folder("data/market_analysis")


# ----------------------------
# Vectorstore creation (one per dataset)
# ----------------------------
def build_vectorstore(chunks, store_path):
    # Creates a FAISS vectorstore from chunks and saves locally
    embeddings = get_embeddings()
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vs.save_local(store_path)
    print(f"Vectorstore saved at: {store_path}")
    return vs


def load_vectorstore(store_path):
    # Loads a previously saved FAISS vectorstore
    embeddings = get_embeddings()
    vs = FAISS.load_local(
        store_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vs


def get_or_create_vectorstore(store_path, loader_func, chunker_func):
    # If store already exists -> load. Else build it fresh.
    if os.path.exists(store_path):
        print(f"Loading existing vectorstore: {store_path}")
        return load_vectorstore(store_path)

    print(f"Creating new vectorstore at: {store_path}")
    raw_docs = loader_func()
    if not raw_docs:
        print(f"No data found for {store_path}. Skipping.")
        return None
    chunks = chunker_func(raw_docs)
    return build_vectorstore(chunks, store_path)


def build_all_vectorstores():
    # Convenience function - builds all three vectorstores at once
    get_or_create_vectorstore(
        "vectorstores/finance_10k", load_finance_10k, chunk_10k_filings
    )
    get_or_create_vectorstore(
        "vectorstores/financial_news", load_financial_news, chunk_financial_news
    )
    get_or_create_vectorstore(
        "vectorstores/market_analysis", load_market_analysis, chunk_market_analysis
    )
    print("All vectorstores ready.")


# ----------------------------
# Retrieval
# ----------------------------
def retrieve_documents(vectorstore, query, k=4, metadata_filter=None):
    # Top-k similarity search with optional metadata filtering
    if vectorstore is None:
        return []
    if metadata_filter:
        return vectorstore.similarity_search(query, k=k, filter=metadata_filter)
    return vectorstore.similarity_search(query, k=k)


def retrieve_from_all_sources(query, k=3):
    # Retrieves from all three vectorstores and combines results
    all_docs = []
    for store_path in [
        "vectorstores/finance_10k",
        "vectorstores/financial_news",
        "vectorstores/market_analysis",
    ]:
        if os.path.exists(store_path):
            vs = load_vectorstore(store_path)
            all_docs.extend(retrieve_documents(vs, query, k=k))
    return all_docs


# ----------------------------
# Retrieval Validation
# ----------------------------
def validate_retrieval(query, docs):
    # Simple validation - checks if any docs were retrieved and basic relevance
    if not docs or len(docs) == 0:
        return False, "No documents retrieved."

    # Basic check: at least one doc should share a keyword with query
    query_words = set(query.lower().split())
    for d in docs:
        doc_words = set(d.page_content.lower().split())
        if query_words & doc_words:
            return True, "Retrieval looks relevant."
    return False, "Retrieved docs do not seem relevant to query."


# ----------------------------
# Prompt + Generation
# ----------------------------
def build_prompt():
    # Structured prompt template for grounded answers
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""You are a Financial Market Intelligence assistant.
Use ONLY the information from the context below to answer the question.
If the answer is not present in the context, say "The provided documents do not contain this information."
Be clear, concise, and ground your answer in the retrieved context.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:""",
    )


def format_context(docs):
    # Joins retrieved docs into one context string
    if not docs:
        return "No context available."
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata if hasattr(d, "metadata") else {}
        meta_str = ", ".join([f"{k}={v}" for k, v in meta.items() if k in
                              ["company", "year", "source", "filing_section",
                               "topic", "analyst_name", "sentiment"]])
        parts.append(f"[Doc {i} | {meta_str}]\n{d.page_content}")
    return "\n\n".join(parts)


def generate_response(query, docs, chat_history=""):
    # Calls the LLM with retrieved context to produce a grounded answer
    llm = get_llm()
    prompt = build_prompt()
    context = format_context(docs)

    final_prompt = prompt.format(
        context=context,
        chat_history=chat_history,
        question=query,
    )
    result = llm.invoke(final_prompt)
    # ChatGoogleGenerativeAI returns an AIMessage object
    return result.content if hasattr(result, "content") else str(result)


# ----------------------------
# RAG as a Tool
# ----------------------------
def rag_tool(query, k=3):
    # This is the callable RAG tool that agents can use
    docs = retrieve_from_all_sources(query, k=k)
    is_valid, msg = validate_retrieval(query, docs)

    if not is_valid:
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": [],
            "validation": msg,
        }

    answer = generate_response(query, docs)
    sources = [d.page_content[:200] for d in docs]
    return {
        "answer": answer,
        "sources": sources,
        "validation": msg,
    }


# ----------------------------
# Full pipeline execution (entry point)
# ----------------------------
def run_pipeline(query, chat_history=""):
    # Main entry point - runs the full RAG flow end-to-end
    docs = retrieve_from_all_sources(query, k=3)
    is_valid, msg = validate_retrieval(query, docs)

    if not is_valid:
        return {
            "response": "Sorry, I could not find relevant information.",
            "context": [],
            "validation": msg,
        }

    response = generate_response(query, docs, chat_history)
    return {
        "response": response,
        "context": [d.page_content for d in docs],
        "validation": msg,
    }


if __name__ == "__main__":
    # Quick test - build vectorstores and run a sample query
    build_all_vectorstores()
    result = run_pipeline("What are the biggest risk factors mentioned for Tesla?")
    print("\nAnswer:\n", result["response"])
