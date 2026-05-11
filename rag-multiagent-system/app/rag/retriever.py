# Retrieval helpers - similarity search, top-k retrieval, metadata filtering
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_store(store_path):
    embeddings = get_embeddings()
    return FAISS.load_local(
        store_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def similarity_search(store_path, query, k=4):
    # Basic top-k similarity search
    if not os.path.exists(store_path):
        return []
    vs = load_store(store_path)
    return vs.similarity_search(query, k=k)


def metadata_filtered_search(store_path, query, metadata_filter, k=4):
    # Search filtered by metadata
    # e.g. metadata_filter = {"filing_section": "Risk Factors"}
    if not os.path.exists(store_path):
        return []
    vs = load_store(store_path)
    return vs.similarity_search(query, k=k, filter=metadata_filter)


# ----------------------------
# Convenience functions for each dataset
# ----------------------------
def retrieve_10k(query, k=4, section=None):
    if section:
        return metadata_filtered_search(
            "vectorstores/finance_10k", query, {"filing_section": section}, k
        )
    return similarity_search("vectorstores/finance_10k", query, k)


def retrieve_news(query, k=4, topic=None):
    if topic:
        return metadata_filtered_search(
            "vectorstores/financial_news", query, {"topic": topic}, k
        )
    return similarity_search("vectorstores/financial_news", query, k)


def retrieve_market_analysis(query, k=4, sentiment=None):
    if sentiment:
        return metadata_filtered_search(
            "vectorstores/market_analysis", query, {"sentiment": sentiment}, k
        )
    return similarity_search("vectorstores/market_analysis", query, k)
