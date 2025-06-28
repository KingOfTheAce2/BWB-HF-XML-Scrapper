"""Clean BWB dataset by stripping XML tags and upload to HF."""

import os
import re
import json
from datasets import load_dataset
from huggingface_hub import HfApi, login

SOURCE_DATASET = "vGassen/Dutch-Basisbestandwetten-Legislation-Laws"
TARGET_DATASET = "vGassen/Dutch-Basisbestandwetten-Legislation-Laws-XML-Clean"

# Only keep documents whose type matches one of the allowed types.  The dataset
# contains many different kinds of regulations.  By default no filtering is
# applied, but the constant below can be customised to restrict the output to
# specific document types.  The value should contain lower‑case strings as they
# appear in the dataset under either the ``type`` or ``document_type`` field.
ALLOWED_TYPES = {
    "ambv",
    "ministeriele-regeling",
    "kb",
    "wet",
}
CHUNK_SIZE = 1000


def strip_xml(text: str) -> str:
    """Remove XML tags from a string."""
    return re.sub(r"<[^>]+>", "", text)


def main() -> None:
    # Stream the dataset to avoid storing the raw XML locally
    dataset = load_dataset(SOURCE_DATASET, split="train", streaming=True)

    token = os.environ.get("HF_TOKEN")
    api = HfApi()
    if token:
        login(token=token)
    api.create_repo(repo_id=TARGET_DATASET, repo_type="dataset", exist_ok=True)

    buffer = []
    shard_idx = 0

    skip_suffixes = ("manifest.xml", ".WTI")

    for record in dataset:
        url = record.get("url") or ""
        if url.lower().endswith(skip_suffixes):
            continue

        doc_type = (record.get("type") or record.get("document_type") or "").lower()
        if ALLOWED_TYPES and doc_type and doc_type not in ALLOWED_TYPES:
            continue

        raw_text = record.get("content") or record.get("text", "")
        if not raw_text.strip():
            continue
        if not raw_text.lstrip().startswith("<"):
            # Skip non-XML records such as index entries.
            continue

        buffer.append(
            {
                "url": url,
                "content": strip_xml(raw_text),
                "source": "Basiswettenbestand",
            }
        )

        if len(buffer) >= CHUNK_SIZE:
            file_path = f"data_{shard_idx:05d}.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for row in buffer:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=TARGET_DATASET,
                repo_type="dataset",
            )
            os.remove(file_path)
            shard_idx += 1
            buffer = []

    if buffer:
        file_path = f"data_{shard_idx:05d}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for row in buffer:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,
            repo_id=TARGET_DATASET,
            repo_type="dataset",
        )
        os.remove(file_path)


if __name__ == "__main__":
    main()

