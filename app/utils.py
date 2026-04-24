from pathlib import Path
import json
import pandas as pd
from langchain_core.documents import Document


def load_text_file(file_path: Path, metadata: dict) -> Document:
    content = file_path.read_text(encoding="utf-8")
    return Document(page_content=content, metadata=metadata)


def load_json_file(file_path: Path, metadata: dict) -> Document:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    content = json.dumps(payload, indent=2)
    merged_metadata = {**metadata, **payload}
    return Document(page_content=content, metadata=merged_metadata)


def load_csv_file(file_path: Path, metadata: dict) -> Document:
    df = pd.read_csv(file_path)
    content = df.to_markdown(index=False)

    return Document(
        page_content=content,
        metadata={
            **metadata,
            "row_count": len(df),
            "columns": list(df.columns),
        },
    )


def load_pdf_file(file_path: Path, metadata: dict) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    for i, doc in enumerate(docs):
        doc.metadata.update(
            {
                **metadata,
                "page_number": i + 1,
                "filename": file_path.name,
                "source": str(file_path),
            }
        )

    return docs