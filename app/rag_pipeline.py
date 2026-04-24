from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import RAW_DATA_DIR, FAISS_INDEX_DIR, EMBEDDING_MODEL
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
    }

    all_documents = []

    for folder_name, document_type in folder_mapping.items():
        folder_docs = load_documents_from_folder(
            base_path / folder_name,
            document_type
        )
        all_documents.extend(folder_docs)

    return all_documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    return chunks


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def build_faiss_index() -> None:
    documents = load_all_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)

    print(f"FAISS index saved to: {FAISS_INDEX_DIR}")


def load_faiss_index() -> FAISS:
    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


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
    build_faiss_index()

    test_search(
        "Is sudden pipe burst water damage covered and what exclusions apply?"
    )

    test_search(
        "Does the customer have any prior claims?"
    )

    test_search(
        "What is the estimated repair cost for claim CLM-2001?"
    )