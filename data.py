"""
data.py - Data loading, sampling, train/test split, and encoding.

Downloads UCSD Goodreads reviews by genre, samples a subset,
splits into train/test, tokenizes with DistilBERT, and returns
ready-to-use PyTorch datasets.
"""

import gzip
import json
import random
import requests

from transformers import DistilBertTokenizerFast

from utils import build_label_maps, ReviewDataset


# URLs for each genre in the UCSD Goodreads dataset.
GENRE_URL_DICT = {
    "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


def load_reviews(url, head=10000, sample_size=2000):
    """
    Stream reviews from a gzipped JSON URL and return a random sample.

    Args:
        url: URL to a .json.gz file with one JSON object per line.
        head: maximum number of reviews to read from the file.
        sample_size: number of reviews to randomly sample from those read.

    Returns:
        list of review text strings.
    """
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def download_all_genres(genre_url_dict=None, head=10000, sample_size=2000):
    """
    Download and sample reviews for every genre.

    Returns:
        dict mapping genre name -> list of review strings.
    """
    if genre_url_dict is None:
        genre_url_dict = GENRE_URL_DICT

    genre_reviews = {}
    for genre, url in genre_url_dict.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews[genre] = load_reviews(url, head=head, sample_size=sample_size)
    return genre_reviews


def split_data(genre_reviews_dict, reviews_per_genre=1000, train_ratio=0.8):
    """
    Split genre reviews into train and test sets.

    Args:
        genre_reviews_dict: dict mapping genre -> list of review strings.
        reviews_per_genre: how many reviews to use per genre.
        train_ratio: fraction of reviews_per_genre used for training.

    Returns:
        (train_texts, train_labels, test_texts, test_labels) as lists.
    """
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    split_point = int(reviews_per_genre * train_ratio)

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(reviews_per_genre, len(reviews)))
        for review in sampled[:split_point]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[split_point:]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


def encode_and_build_datasets(
    train_texts, train_labels, test_texts, test_labels, model_name="distilbert-base-cased", max_length=512
):
    """
    Tokenize texts and build PyTorch datasets.

    Args:
        train_texts: list of training review strings.
        train_labels: list of training genre labels.
        test_texts: list of test review strings.
        test_labels: list of test genre labels.
        model_name: HuggingFace model identifier for the tokenizer.
        max_length: maximum token length for truncation/padding.

    Returns:
        (train_dataset, test_dataset, label2id, id2label, tokenizer)
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    label2id, id2label = build_label_maps(train_labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_labels_enc = [label2id[y] for y in train_labels]
    test_labels_enc = [label2id[y] for y in test_labels]

    train_dataset = ReviewDataset(train_encodings, train_labels_enc)
    test_dataset = ReviewDataset(test_encodings, test_labels_enc)

    return train_dataset, test_dataset, label2id, id2label, tokenizer


def prepare_data(model_name="distilbert-base-cased", max_length=512, reviews_per_genre=1000):
    """
    End-to-end data preparation: download, split, encode.

    Args:
        model_name: HuggingFace model identifier.
        max_length: max token length.
        reviews_per_genre: number of reviews to sample per genre.

    Returns:
        (train_dataset, test_dataset, label2id, id2label, tokenizer,
         train_labels, test_labels)
    """
    genre_reviews = download_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(
        genre_reviews, reviews_per_genre=reviews_per_genre
    )

    print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")

    train_dataset, test_dataset, label2id, id2label, tokenizer = encode_and_build_datasets(
        train_texts, train_labels, test_texts, test_labels, model_name=model_name, max_length=max_length
    )

    return train_dataset, test_dataset, label2id, id2label, tokenizer, train_labels, test_labels


if __name__ == "__main__":
    train_ds, test_ds, l2id, id2l, tok, _, _ = prepare_data()
    print(f"Labels: {l2id}")
    print(f"Train dataset length: {len(train_ds)}")
    print(f"Test dataset length: {len(test_ds)}")
