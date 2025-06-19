"""Clean BWB dataset by stripping XML tags and upload to HF."""

import os
import re
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login

SOURCE_DATASET = "vGassen/Dutch-Basisbestandwetten-Legislation-Laws"
TARGET_DATASET = "vGassen/Dutch-Basisbestandwetten-Legislation-Laws-XML-Clean"


def strip_xml(text: str) -> str:
    """Remove XML tags from a string."""
    return re.sub(r"<[^>]+>", "", text)


def main() -> None:
    # Stream the dataset to avoid storing the raw XML locally
    dataset = load_dataset(SOURCE_DATASET, split="train", streaming=True)

    def record_generator():
        for record in dataset:
            raw_text = record.get("content") or record.get("text", "")
            yield {
                "url": record.get("url"),
                "content": strip_xml(raw_text),
                "source": "Basiswettenbestand",
            }

    cleaned_dataset = Dataset.from_generator(record_generator)

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    HfApi().create_repo(repo_id=TARGET_DATASET, repo_type="dataset", exist_ok=True)
    cleaned_dataset.push_to_hub(TARGET_DATASET, private=False)


if __name__ == "__main__":
    main()