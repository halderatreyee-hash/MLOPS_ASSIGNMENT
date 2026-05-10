"""
utils.py - Shared helpers for the MLOps pipeline.

Contains label map construction, custom PyTorch Dataset class,
and the compute_metrics function used during training and evaluation.
"""

import torch
from sklearn.metrics import accuracy_score, f1_score


def build_label_maps(labels):
    """
    Build label-to-id and id-to-label mappings from a list of labels.

    Args:
        labels: list of string labels (e.g. genre names).

    Returns:
        label2id: dict mapping label string -> integer id.
        id2label: dict mapping integer id -> label string.
    """
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


class ReviewDataset(torch.utils.data.Dataset):
    """
    A simple PyTorch Dataset that wraps HuggingFace tokenizer encodings
    and integer-encoded labels.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """
    Compute accuracy and weighted F1 from a Trainer prediction object.

    Args:
        pred: transformers.EvalPrediction with .label_ids and .predictions.

    Returns:
        dict with 'accuracy' and 'f1' keys.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }
