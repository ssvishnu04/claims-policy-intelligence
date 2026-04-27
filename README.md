![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-purple)

# Claims & Policy Intelligence Platform
An AI-powered Claims & Policy Intelligence Platform for Property & Casualty (P&C) insurance, enabling adjusters and underwriters to query coverage, exclusions, and claim details using Retrieval-Augmented Generation (RAG).

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

## Architecture

User Query (Streamlit UI)  
↓  
Claim Context Input (Policy ID, Claim ID)  
↓  
RAG Pipeline  
- Structured Data Layer (FNOL, Claims, Estimates)  
- FAISS Vector Search (Policies, Notes, Guidelines)  
- LangChain Orchestration  
↓  
Groq LLM (Response Generation)  
↓  
AI Answer + Sources (Explainable Output)

## RAGAS Evaluation Results

| Metric | Score |
|---|---:|
| Context Precision | **1.0000** |
| Context Recall | **0.9000** |
| Faithfulness | **0.8768** |
| Answer Relevancy | **0.9600** |

## Key Design Decisions

- Implemented strict claim-level filtering to prevent cross-claim data leakage
- Prioritized structured data over retrieved documents for claim facts
- Enforced grounding rules to reduce hallucination
- Increased retrieval depth (k=20) to improve recall

## Sample Queries

- Is this claim covered?
- What exclusions apply?
- What deductible should be reviewed?
- What supporting documents are relevant?
- What should the adjuster verify next?

## Streamlit Demo

https://claims-policy-assistant.streamlit.app/

<p align="center">
  <img src="image/PolicyClaimAssistant.jpg" width="900"/>
</p>

## Business Impact

- Reduced claim review time through instant policy lookup
- Improved decision accuracy using explainable responses
- Eliminated manual document search for adjusters
