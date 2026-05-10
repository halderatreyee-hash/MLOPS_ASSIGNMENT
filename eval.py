"""
eval.py - Evaluation, metrics reporting, W&B artifact upload, and HuggingFace Hub push.

Loads the fine-tuned model saved by train.py, runs evaluation on the
test set, logs final metrics to W&B, saves a classification report as
a W&B artifact, and optionally pushes the model to Hugging Face Hub.
"""

import argparse
import json
import os
import pickle

import wandb
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import prepare_data
from utils import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model and push to HF Hub.")
    parser.add_argument("--model_path", type=str, default="./results/final_model",
                        help="Path to the saved fine-tuned model directory.")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--model_name", type=str, default="distilbert-base-cased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--reviews_per_genre", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model and tokenizer to Hugging Face Hub.")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="HF Hub repo id, e.g. 'your-username/distilbert-goodreads-genres'.")
    parser.add_argument("--wandb_project", type=str, default="mlops-assignment2")
    parser.add_argument("--run_name", type=str, default="distilbert-eval")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load saved artifacts from training ---
    artifacts_path = os.path.join(args.output_dir, "data_artifacts.pkl")
    if os.path.exists(artifacts_path):
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
        label2id = artifacts["label2id"]
        id2label = artifacts["id2label"]
        saved_test_labels = artifacts["test_labels"]
    else:
        label2id, id2label, saved_test_labels = None, None, None

    # --- Prepare data (re-downloads; in production you would cache) ---
    print("Preparing data...")
    train_dataset, test_dataset, label2id_new, id2label_new, tokenizer, _, test_labels = prepare_data(
        model_name=args.model_name,
        max_length=args.max_length,
        reviews_per_genre=args.reviews_per_genre,
    )

    # Prefer saved label maps for consistency with the trained model
    if label2id is not None:
        id2label_eval = id2label
    else:
        id2label_eval = id2label_new

    if saved_test_labels is not None:
        test_labels = saved_test_labels

    # --- Load fine-tuned model ---
    print(f"Loading model from {args.model_path}...")
    model = DistilBertForSequenceClassification.from_pretrained(args.model_path)

    # --- Initialise W&B ---
    wandb.init(project=args.wandb_project, name=args.run_name)

    # --- Build a Trainer for evaluation ---
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Run evaluation ---
    print("Running evaluation...")
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # --- Log final metrics to W&B ---
    wandb.log({
        "final/loss": eval_results["eval_loss"],
        "final/accuracy": eval_results["eval_accuracy"],
        "final/f1": eval_results["eval_f1"],
    })

    # --- Classification report ---
    preds = trainer.predict(test_dataset).predictions.argmax(-1)
    labels = [item["labels"].item() for item in test_dataset]

    report = classification_report(
        labels, preds,
        target_names=list(id2label_eval.values()),
        output_dict=True,
    )

    report_path = os.path.join(args.output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Classification report saved to {report_path}")

    # --- Upload report as a W&B artifact ---
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(report_path)
    wandb.log_artifact(artifact)

    # --- Optionally push to Hugging Face Hub ---
    if args.push_to_hub and args.hf_repo:
        print(f"Pushing model and tokenizer to HF Hub: {args.hf_repo}")
        model.push_to_hub(args.hf_repo)
        tokenizer.push_to_hub(args.hf_repo)
        hf_url = f"https://huggingface.co/{args.hf_repo}"
        wandb.run.summary["huggingface_model"] = hf_url
        print(f"Model published at {hf_url}")

    wandb.finish()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
