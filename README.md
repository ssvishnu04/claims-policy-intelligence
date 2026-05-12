![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-purple)
![Azure](https://img.shields.io/badge/Azure-Cloud-blue)
![Databricks](https://img.shields.io/badge/Databricks-Lakehouse-red)

# P&C Claims Intelligence RAG Platform

Enterprise-style AI-powered Claims & Policy Intelligence Platform for Property & Casualty (P&C) insurance enabling adjusters and underwriters to retrieve coverage details, exclusions, deductible information, and claim insights using Retrieval-Augmented Generation (RAG).

---

# Executive Summary

Insurance claims teams often spend significant time manually reviewing policies, FNOL reports, adjuster notes, and underwriting documents to validate coverage and determine next steps during claims adjudication.

This platform demonstrates how Generative AI and Retrieval-Augmented Generation (RAG) can be applied within a regulated P&C insurance environment to improve claims investigation efficiency, accelerate policy lookup, and generate explainable AI-assisted responses grounded in enterprise insurance documents.

The solution combines:
- Structured claims data
- Semantic vector search
- LLM-powered reasoning
- Governance-aware retrieval workflows
- RAGAS evaluation metrics
- Enterprise-style API orchestration

---

# Business Use Case

This application helps:
- Claims Adjusters
- Underwriters
- SIU Investigators
- Claims Operations Teams

retrieve critical insurance information from:
- Policy documents
- FNOL reports
- Adjuster notes
- Underwriting guidelines
- Supporting claim documents

The system enables faster claims review and reduces manual document search effort while improving response explainability.

---

# Enterprise Features

- Retrieval-Augmented Generation (RAG)
- Semantic Vector Search using FAISS
- Governance-aware claim filtering
- Explainable AI responses
- Structured + unstructured data integration
- LangChain orchestration workflows
- REST API integration
- RAGAS evaluation framework
- Streamlit-based UI
- Cloud-ready deployment architecture

---

# Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | Streamlit |
| LLM Orchestration | LangChain |
| Embeddings | Hugging Face |
| Vector Search | FAISS |
| LLM Provider | Groq API |
| Evaluation | RAGAS |
| APIs | Python / Flask |
| Cloud & Processing | Azure + Databricks |
| Data Processing | PySpark |

---

# Key Concepts

## FNOL (First Notice of Loss)
The initial report of a claim submitted by the insured after a loss event.

## RAG (Retrieval-Augmented Generation)
Combines enterprise document retrieval with Large Language Model (LLM) response generation to produce grounded and explainable answers.

## Vector Search
Uses embeddings and semantic similarity to retrieve relevant insurance documents and claim context.

---

# High-Level Architecture

```text
User Query (Streamlit UI)
        ↓
Claim Context Validation
(Policy ID / Claim ID)
        ↓
RAG Orchestration Layer
        ├── Structured Claims Data
        ├── Policy Documents
        ├── Adjuster Notes
        ├── Underwriting Guidelines
        ↓
FAISS Semantic Vector Search
        ↓
LangChain Retrieval Workflow
        ↓
Groq LLM Response Generation
        ↓
Governed AI Response + Sources
```

---

# Governance & AI Controls

The platform incorporates governance-aware retrieval workflows designed for regulated insurance environments:

- Claim-level filtering to prevent cross-claim data leakage
- Context grounding rules to reduce hallucinations
- Structured data prioritization for critical claim facts
- Retrieval depth optimization for improved recall
- Explainable source-backed responses
- Controlled document retrieval pipelines

---

# RAGAS Evaluation Results

| Metric | Score |
|---|---:|
| Context Precision | **1.0000** |
| Context Recall | **0.9000** |
| Faithfulness | **0.8768** |
| Answer Relevancy | **0.9600** |

---

# Repository Structure

```text
project/
│
├── app/
├── api/
├── data/
├── vectorstore/
├── evaluation/
├── ingestion/
├── pipelines/
├── screenshots/
├── requirements.txt
├── streamlit_app.py
└── README.md
```

---

# Sample Queries

- Is this claim covered?
- What exclusions apply?
- What deductible should be reviewed?
- What supporting documents are relevant?
- What should the adjuster verify next?
- What underwriting risks should be considered?

---

# Streamlit Demo

https://claims-policy-assistant.streamlit.app/

<p align="center">
  <img src="image/PolicyClaimAssistant.jpg" width="900"/>
</p>

---

# Business Impact

- Accelerated policy and claim document review workflows
- Reduced manual document search effort for adjusters
- Improved explainability through grounded AI responses
- Enabled faster claims triage and investigation support
- Demonstrated enterprise GenAI adoption patterns for regulated insurance environments

---

# Future Enhancements

- Multi-agent claims orchestration
- Real-time claim event ingestion
- Azure OpenAI integration
- Human-in-the-loop review workflows
- Advanced governance observability dashboards
- Enterprise authentication & RBAC
- Hybrid vector database integration

---
⭐ Enterprise-style AI solution demonstrating practical RAG implementation patterns for regulated P&C insurance workflows.
