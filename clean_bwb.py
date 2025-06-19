import re
from datasets import load_dataset, DatasetDict
from datasets import Dataset
import os
from huggingface_hub import HfApi, login


def strip_xml(text: str) -> str:
    """Remove XML tags from text."""
    # simple regex to drop tags
    return re.sub(r"<[^>]+>", "", text)


def main():
    # load original dataset
    dataset = load_dataset("vGassen/Dutch-Basisbestandwetten-Legislation-Laws", split="train")

    # transform dataset by stripping XML tags
    cleaned_content = [strip_xml(row["content"]) for row in dataset]

    cleaned_dataset = Dataset.from_dict({
        "url": dataset["url"],
        "content": cleaned_content,
        "source": ["Basiswettenbestand"] * len(dataset),
    })

    cleaned_dataset = DatasetDict({"train": cleaned_dataset})

    # push to new repository
    repo_id = "vGassen/Dutch-Basisbestandwetten-Legislation-Laws-XML-Clean"
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    cleaned_dataset.push_to_hub(repo_id)


if __name__ == "__main__":
    main()
