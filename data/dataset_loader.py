from sklearn.datasets import fetch_20newsgroups
import re


def clean_text(text: str) -> str:
    text = re.sub(r"(From|Subject|Organization|Lines):.*", "", text)
    text = re.sub(r">.*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset():
    dataset = fetch_20newsgroups(remove=("headers", "footers", "quotes"))
    docs = [clean_text(d) for d in dataset.data]
    return docs, dataset.target