"""
train.py - Model loading, Trainer setup, and training loop.

Loads a pre-trained DistilBERT model, configures HuggingFace Trainer
with W&B logging, and runs fine-tuning on the Goodreads genre
classification task.
"""

import argparse
import os
import pickle

import torch
import wandb
from transformers import (
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from data import prepare_data
from utils import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Goodreads genres.")
    parser.add_argument("--model_name", type=str, default="distilbert-base-cased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--reviews_per_genre", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--run_name", type=str, default="distilbert-run-1")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cuda' or 'cpu'. Auto-detected if omitted.")
    return parser.parse_args()


def get_device(preferred=None):
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # --- Data preparation ---
    print("Preparing data...")
    train_dataset, test_dataset, label2id, id2label, tokenizer, train_labels, test_labels = prepare_data(
        model_name=args.model_name,
        max_length=args.max_length,
        reviews_per_genre=args.reviews_per_genre,
    )

    # Save artifacts needed by eval.py
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "data_artifacts.pkl"), "wb") as f:
        pickle.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
                "test_labels": test_labels,
            },
            f,
        )

    # --- Initialise W&B ---
    wandb.init(
        project="mlops-assignment2",
        name=args.run_name,
        config={
            "model": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "dataset": "UCSD Goodreads",
            "reviews_per_genre": args.reviews_per_genre,
        },
    )

    # --- Load model ---
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(id2label)
    ).to(device)

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=args.run_name,
    )

    # --- Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model and tokenizer locally
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print(f"Model saved to {args.output_dir}/final_model")

    # Save trainer state for eval.py to pick up
    with open(os.path.join(args.output_dir, "trainer.pkl"), "wb") as f:
        pickle.dump(
            {
                "model_path": os.path.join(args.output_dir, "final_model"),
                "output_dir": args.output_dir,
            },
            f,
        )

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
