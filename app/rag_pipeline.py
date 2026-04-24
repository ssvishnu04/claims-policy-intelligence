from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import RAW_DATA_DIR, FAISS_INDEX_DIR, EMBEDDING_MODEL
from app.utils import load_text_file, load_json_file


def load_all_documents() -> List[Document]:
    base_path = Path(RAW_DATA_DIR)
    documents = []

    for file in (base_path / "policies").glob("*.*"):
        documents.append(load_text_file(file, {
            "document_type": "policy",
            "source": str(file),
            "filename": file.name,
        }))

    for file in (base_path / "fnol").glob("*.json"):
        documents.append(load_json_file(file, {
            "document_type": "fnol",
            "source": str(file),
            "filename": file.name,
        }))

    for file in (base_path / "adjuster_notes").glob("*.*"):
        documents.append(load_text_file(file, {
            "document_type": "adjuster_note",
            "source": str(file),
            "filename": file.name,
        }))

    for file in (base_path / "underwriting_guidelines").glob("*.*"):
        documents.append(load_text_file(file, {
            "document_type": "uw_guideline",
            "source": str(file),
            "filename": file.name,
        }))

    return documents


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
    vectorstore = FAISS.from_documents(chunks, embeddings)

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


def test_search(query: str, k: int = 3) -> None:
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
        "Is sudden pipe burst water damage covered under the homeowners policy?"
    )