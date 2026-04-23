from pathlib import Path
import json
from langchain_core.documents import Document


def load_text_file(file_path: Path, metadata: dict) -> Document:
    content = file_path.read_text(encoding="utf-8")
    return Document(page_content=content, metadata=metadata)


def load_json_file(file_path: Path, metadata: dict) -> Document:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    content = json.dumps(payload, indent=2)
    merged_metadata = {**metadata, **payload}
    return Document(page_content=content, metadata=merged_metadata)