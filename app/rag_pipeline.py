from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import (
    RAW_DATA_DIR,
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)

from app.utils import (
    load_text_file,
    load_json_file,
    load_csv_file,
    load_pdf_file,
)


def load_documents_from_folder(folder_path: Path, document_type: str) -> List[Document]:
    documents = []

    if not folder_path.exists():
        return documents

    for file in folder_path.glob("*.*"):
        base_metadata = {
            "document_type": document_type,
            "source": str(file),
            "filename": file.name,
        }

        if file.suffix.lower() == ".txt":
            documents.append(load_text_file(file, base_metadata))

        elif file.suffix.lower() == ".json":
            documents.append(load_json_file(file, base_metadata))

        elif file.suffix.lower() == ".csv":
            documents.append(load_csv_file(file, base_metadata))

        elif file.suffix.lower() == ".pdf":
            documents.extend(load_pdf_file(file, base_metadata))

        else:
            print(f"Skipping unsupported file type: {file}")

    return documents


def load_all_documents() -> List[Document]:
    base_path = Path(RAW_DATA_DIR)

    folder_mapping = {
        "policies": "policy",
        "fnol": "fnol",
        "api_data": "api_data",
        "claims_history": "claims_history",
        "adjuster_notes": "adjuster_note",
        "underwriting_guidelines": "uw_guideline",
        "repair_estimates": "repair_estimate",
        "supporting_documents": "supporting_document",
    }

    all_documents = []

    for folder_name, document_type in folder_mapping.items():
        all_documents.extend(
            load_documents_from_folder(base_path / folder_name, document_type)
        )

    return all_documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    return chunks


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_faiss_index() -> None:
    documents = load_all_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)

    print(f"FAISS index saved to: {FAISS_INDEX_DIR}")


def load_faiss_index() -> FAISS:
    embeddings = get_embeddings()

    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def filter_docs_by_policy_claim(
    docs: List[Document],
    policy_id: str,
    claim_id: str,
) -> List[Document]:
    """
    Keeps only chunks related to the selected policy or claim.
    This improves retrieval precision and avoids mixing unrelated claims/policies.
    """
    filtered_docs = []

    policy_id_lower = policy_id.lower().strip()
    claim_id_lower = claim_id.lower().strip()

    for doc in docs:
        content = doc.page_content.lower()
        metadata_text = " ".join(str(value).lower() for value in doc.metadata.values())

        policy_match = policy_id_lower in content or policy_id_lower in metadata_text
        claim_match = claim_id_lower in content or claim_id_lower in metadata_text

        if policy_match or claim_match:
            filtered_docs.append(doc)

    return filtered_docs


def format_context(docs: List[Document]) -> str:
    context_blocks = []

    for i, doc in enumerate(docs, start=1):
        context_blocks.append(
            f"""
SOURCE {i}
Document Type: {doc.metadata.get("document_type")}
File Name: {doc.metadata.get("filename")}
Chunk ID: {doc.metadata.get("chunk_id")}

Content:
{doc.page_content}
"""
        )

    return "\n".join(context_blocks)


def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0,
    )


def ask_claims_assistant(
    question: str,
    policy_id: str,
    claim_id: str,
    k: int = 10,
) -> Dict[str, Any]:

    vectorstore = load_faiss_index()

    scoped_query = f"""
Policy ID: {policy_id}
Claim ID: {claim_id}
Question: {question}
"""

    retrieved_docs = vectorstore.similarity_search(scoped_query, k=k)

    filtered_docs = filter_docs_by_policy_claim(
        docs=retrieved_docs,
        policy_id=policy_id,
        claim_id=claim_id,
    )

    # If strict filtering removes everything, return a safe response.
    if not filtered_docs:
        return {
            "answer": (
                "Decision Summary:\n"
                "I could not find enough matching context for the provided Policy ID and Claim ID.\n\n"
                "Coverage Details:\n"
                "Not enough information available.\n\n"
                "Exclusions / Limitations:\n"
                "Not enough information available.\n\n"
                "Claim or Prior Loss Context:\n"
                "No matching claim or policy context was found.\n\n"
                "Recommended Next Steps:\n"
                "Verify that the Policy ID and Claim ID are correct and rebuild the FAISS index if new documents were added.\n\n"
                "Sources Used:\n"
                "No matching sources found."
            ),
            "sources": [],
        }

    context = format_context(filtered_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an AI Claims & Policy Assistant for a Property & Casualty insurance company.

Rules:
1. Answer only using the retrieved context.
2. Only answer for the provided Policy ID and Claim ID.
3. Do not use unrelated policies, claims, or documents.
4. Do not make up policy terms, exclusions, deductibles, or claim facts.
5. If the context does not support the answer, clearly say more information is needed.
6. Explain the answer in simple insurance business language.
7. Mention relevant source files when useful.

Return your answer in this format:

Decision Summary:
Coverage Details:
Exclusions / Limitations:
Claim or Prior Loss Context:
Recommended Next Steps:
Sources Used:
""",
            ),
            (
                "human",
                """
Policy ID:
{policy_id}

Claim ID:
{claim_id}

Question:
{question}

Retrieved Context:
{context}
""",
            ),
        ]
    )

    chain = prompt | get_llm() | StrOutputParser()

    answer = chain.invoke(
        {
            "policy_id": policy_id,
            "claim_id": claim_id,
            "question": question,
            "context": context,
        }
    )

    return {
        "answer": answer,
        "sources": filtered_docs,
    }


def test_search(query: str, k: int = 4) -> None:
    vectorstore = load_faiss_index()
    results = vectorstore.similarity_search(query, k=k)

    print("\nSearch Query:")
    print(query)
    print("=" * 80)

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("-" * 80)
        print("Metadata:")
        print(doc.metadata)
        print("\nContent:")
        print(doc.page_content[:700])


if __name__ == "__main__":
    result = ask_claims_assistant(
        policy_id="POL-1001",
        claim_id="CLM-2001",
        question="Is this sudden pipe burst water damage covered and what exclusions apply?",
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(source.metadata)