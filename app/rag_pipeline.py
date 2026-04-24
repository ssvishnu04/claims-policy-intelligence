from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.config import RAW_DATA_DIR
from app.utils import (
    load_text_file,
    load_json_file,
    load_csv_file,
    load_pdf_file,
)


def load_documents_from_folder(
    folder_path: Path,
    document_type: str
) -> List[Document]:
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


if __name__ == "__main__":
    docs = load_all_documents()
    print(f"Loaded {len(docs)} documents\n")

    for doc in docs:
        print("=" * 80)
        print("Metadata:")
        print(doc.metadata)
        print("\nContent Preview:")
        print(doc.page_content[:700])
        print()