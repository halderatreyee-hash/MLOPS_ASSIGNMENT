# DistilBERT Goodreads Genre Classification

Fine-tuning DistilBERT for multi-class text classification on the UCSD Goodreads book review dataset. Reviews are classified into eight genres: poetry, children, comics/graphic, fantasy/paranormal, history/biography, mystery/thriller/crime, romance, and young adult. The project follows an end-to-end MLOps workflow with experiment tracking via Weights & Biases and model hosting on Hugging Face Hub.

## Project Structure

```
.
├── data.py            # Data downloading, sampling, train/test split, tokenization
├── train.py           # Model loading, Trainer setup, training loop with W&B logging
├── eval.py            # Evaluation, metrics, classification report, HF Hub push
├── utils.py           # Shared helpers (label maps, dataset class, compute_metrics)
├── requirements.txt   # Python dependencies
└── README.md
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mlops-assignment2.git
   cd mlops-assignment2
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Log in to Weights & Biases and Hugging Face:
   ```bash
   wandb login
   huggingface-cli login
   ```

## Usage

### Training

Run the training script. On CPU, reduce `--reviews_per_genre` to keep training time manageable:

```bash
# GPU (recommended, e.g. Google Colab)
python train.py --device cuda

# CPU (smaller dataset for faster iteration)
python train.py --device cpu --reviews_per_genre 200
```

Key arguments:

| Argument              | Default               | Description                       |
|-----------------------|-----------------------|-----------------------------------|
| --model_name          | distilbert-base-cased | HuggingFace model identifier      |
| --epochs              | 3                     | Number of training epochs         |
| --train_batch_size    | 16                    | Training batch size per device    |
| --learning_rate       | 3e-5                  | Learning rate                     |
| --reviews_per_genre   | 1000                  | Reviews sampled per genre         |

### Evaluation

After training, run evaluation and optionally push to Hugging Face Hub:

```bash
# Evaluate only
python eval.py

# Evaluate and push to HF Hub
python eval.py --push_to_hub --hf_repo your-username/distilbert-goodreads-genres
```

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.205 |
| F1 Score  | 0.133 |
| Eval Loss | 2.054 |

## Links

- Hugging Face model: https://huggingface.co/Atreyee-Halder/distilbert-goodreads-genres
- W&B dashboard: https://wandb.ai/g25ait2023-iit-jodhpur/mlops-assignment2
