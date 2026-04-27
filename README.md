# Claims & Policy Intelligence Platform

A portfolio project for Property & Casualty insurance demonstrating an end-to-end Retrieval-Augmented Generation (RAG) application using LangChain, FAISS, OpenAI, and Streamlit.

## Business Use Case
This application helps claims adjusters and underwriters quickly retrieve coverage details, exclusions, deductible information, and prior claim context from insurance documents such as policies, FNOL reports, adjuster notes, and underwriting guidelines.

## Tech Stack
- Python
- Streamlit
- LangChain
- Hugging Face (Embeddings)
- FAISS (Vector Search)
- Groq API (LLM)
- RAGAS

## Key Concepts

- **FNOL (First Notice of Loss):** The initial report of a claim submitted by the insured.
- **RAG (Retrieval-Augmented Generation):** Combines document retrieval with LLM responses.
- **Vector Search:** Enables semantic search using embeddings.

## Architecture (High Level)

User Query (Streamlit UI) ---> Claim Context Input (Policy ID, Claim ID) ---> RAG Pipeline (Structured Data Layer (FNOL, Claims, Estimates) ||  Vector Search (FAISS Index) || LangChain Orchestration)--->
Groq LLM (Response Generation) ---> Internal Platform Answer + Sources(Explainable Output)

## RAGAS Evaluation Results

Metric	                          Score
Context Precision	         1.0000
Context Recall	                 0.9000
Faithfulness	                 0.8768
Answer Relevancy	         0.9600

## Streamlit Demo

![Claims & Policy Intelligence Platform](image/PolicyClaimAssistant.jpg)
