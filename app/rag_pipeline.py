from pathlib import Path
from typing import List
from langchain_core.documents import Document

from app.config import RAW_DATA_DIR
from app.utils import load_text_file, load_json_file


def load_all_documents() -> List[Document]:
    base_path = Path(RAW_DATA_DIR)
    documents = []

    # Policies
    for file in (base_path / "policies").glob("*.*"):
        documents.append(
            load_text_file(
                file,
                {
                    "document_type": "policy",
                    "source": str(file),
                    "filename": file.name
                }
            )
        )

    # FNOL
    for file in (base_path / "fnol").glob("*.json"):
        documents.append(
            load_json_file(
                file,
                {
                    "document_type": "fnol",
                    "source": str(file),
                    "filename": file.name
                }
            )
        )

    # Adjuster Notes
    for file in (base_path / "adjuster_notes").glob("*.*"):
        documents.append(
            load_text_file(
                file,
                {
                    "document_type": "adjuster_note",
                    "source": str(file),
                    "filename": file.name
                }
            )
        )

    # Underwriting Guidelines
    for file in (base_path / "underwriting_guidelines").glob("*.*"):
        documents.append(
            load_text_file(
                file,
                {
                    "document_type": "uw_guideline",
                    "source": str(file),
                    "filename": file.name
                }
            )
        )

    return documents


if __name__ == "__main__":
    docs = load_all_documents()
    print(f"Loaded {len(docs)} documents\n")

    for doc in docs:
        print("=" * 80)
        print(doc.metadata)
        print(doc.page_content[:500])
        print()