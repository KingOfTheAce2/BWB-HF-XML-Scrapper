"""Clean BWB dataset by stripping XML tags and upload to HF."""

import os
import re
import json
from datasets import load_dataset
from huggingface_hub import HfApi, login

SOURCE_DATASET = "vGassen/Dutch-Basisbestandwetten-Legislation-Laws"
TARGET_DATASET = "vGassen/Dutch-Basisbestandwetten-Legislation-Laws-XML-Clean"
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

    for record in dataset:
        raw_text = record.get("content") or record.get("text", "")
        buffer.append(
            {
                "url": record.get("url"),
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