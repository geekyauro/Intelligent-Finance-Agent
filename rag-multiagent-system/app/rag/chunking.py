# Chunking strategies for all three datasets
# All three use metadata-aware chunking (as per the prompt requirement)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# ----------------------------
# 1. Metadata-aware chunking for 10-K filings
# ----------------------------
def chunk_10k_filings(docs, company="Tesla", year="2024"):
    # 10-K filings are long, so we use larger chunks
    # We also try to detect common section headers and attach them as metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Known 10-K section names that we try to match in the text
    known_sections = [
        "Risk Factors",
        "Business",
        "Management Discussion",
        "Financial Statements",
        "Properties",
        "Legal Proceedings",
        "Market Risk",
    ]

    chunks = splitter.split_documents(docs)

    # Add metadata to every chunk
    enriched = []
    for c in chunks:
        section = "General"
        for s in known_sections:
            if s.lower() in c.page_content.lower():
                section = s
                break

        c.metadata.update({
            "company": company,
            "year": year,
            "document_type": "10-K",
            "filing_section": section,
        })
        enriched.append(c)

    print(f"Created {len(enriched)} chunks from 10-K filings")
    return enriched


# ----------------------------
# 2. Metadata-aware chunking for Financial News
# ----------------------------
def chunk_financial_news(docs, company="Tesla"):
    # News articles are shorter, so smaller chunks work better
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    enriched = []
    for c in chunks:
        # Use source filename if available
        source_file = c.metadata.get("source", "unknown")
        topic = "General News"
        # Simple topic detection by keywords
        text_lower = c.page_content.lower()
        if "earnings" in text_lower:
            topic = "Earnings"
        elif "delivery" in text_lower or "deliveries" in text_lower:
            topic = "Deliveries"
        elif "robotaxi" in text_lower or "fsd" in text_lower:
            topic = "Autonomous Driving"

        c.metadata.update({
            "company": company,
            "document_type": "Financial News",
            "source": source_file,
            "topic": topic,
        })
        enriched.append(c)

    print(f"Created {len(enriched)} chunks from financial news")
    return enriched


# ----------------------------
# 3. Metadata-aware chunking for Market Analysis / Analyst Reports
# ----------------------------
def chunk_market_analysis(docs, company="Tesla"):
    # Analyst reports are medium-length structured documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    enriched = []
    for c in chunks:
        # Simple sentiment detection
        text_lower = c.page_content.lower()
        sentiment = "Neutral"
        if any(w in text_lower for w in ["bullish", "buy", "strong", "growth"]):
            sentiment = "Bullish"
        elif any(w in text_lower for w in ["bearish", "sell", "weak", "decline"]):
            sentiment = "Bearish"

        c.metadata.update({
            "company": company,
            "document_type": "Market Analysis",
            "report_type": "Equity Research",
            "sentiment": sentiment,
        })
        enriched.append(c)

    print(f"Created {len(enriched)} chunks from market analysis reports")
    return enriched
