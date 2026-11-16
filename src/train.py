"""Optuna Hyperparameter Optimization + MLflow Experiment Tracking for EmailTextCNN."""
import argparse
import json
import os
import sys

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_dataset import CLASSES, build_dataset
from src.model import EmailTextCNN
from src.tokenizer import EmailTokenizer

ARTIFACTS_DIR = "artifacts"


def load_or_create_data(data_path: str = "data/emails_dataset.json"):
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        records = build_dataset(samples_per_class=240)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            records = json.load(f)

    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.int64)

    split_idx = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]

    tokenizer = EmailTokenizer(max_vocab_size=5000, max_seq_len=48)
    tokenizer.fit(train_texts)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    tokenizer.save(os.path.join(ARTIFACTS_DIR, "tokenizer.json"))

    X_train = tokenizer.encode_batch(train_texts)
    X_val = tokenizer.encode_batch(val_texts)
    return X_train, y_train, X_val, y_val, tokenizer


def train_and_eval(params: dict, X_train, y_train, X_val, y_val, vocab_size: int, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmailTextCNN(
        vocab_size=vocab_size,
        embed_dim=params["embed_dim"],
        num_filters=params["num_filters"],
        num_classes=len(CLASSES),
        dropout=params["dropout"],
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    val_x = torch.tensor(X_val, device=device)
    val_y = torch.tensor(y_val, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            preds = torch.argmax(logits, dim=1)
            val_acc = (preds == val_y).float().mean().item()
            best_val_acc = max(best_val_acc, val_acc)

    return model, best_val_acc


def run_optuna_study(trials: int = 8, epochs: int = 10):
    X_train, y_train, X_val, y_val, tokenizer = load_or_create_data()
    vocab_size = len(tokenizer.word2idx)

    mlflow.set_experiment("Email-Classification-TextCNN")

    best_state = {"acc": -1.0, "model": None, "params": None}

    with mlflow.start_run(run_name="Optuna_Hyperparameter_Search"):
        def objective(trial: optuna.Trial) -> float:
            params = {
                "embed_dim": trial.suggest_categorical("embed_dim", [32, 64, 128]),
                "num_filters": trial.suggest_categorical("num_filters", [32, 64, 128]),
                "dropout": trial.suggest_float("dropout", 0.15, 0.5),
                "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            }

            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                model, val_acc = train_and_eval(params, X_train, y_train, X_val, y_val, vocab_size, epochs)
                mlflow.log_metric("val_accuracy", val_acc)

                if val_acc > best_state["acc"]:
                    best_state["acc"] = val_acc
                    best_state["model"] = model
                    best_state["params"] = params

                return val_acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)

        mlflow.log_params({f"best_{k}": v for k, v in best_state["params"].items()})
        mlflow.log_metric("best_val_accuracy", best_state["acc"])

        # Save local checkpoint & log PyTorch artifact to MLflow
        ckpt_path = os.path.join(ARTIFACTS_DIR, "best_email_cnn.pt")
        torch.save(
            {
                "model_state_dict": best_state["model"].state_dict(),
                "params": best_state["params"],
                "vocab_size": vocab_size,
                "classes": CLASSES,
            },
            ckpt_path,
        )
        mlflow.log_artifact(ckpt_path)
        mlflow.pytorch.log_model(best_state["model"], artifact_path="email_textcnn_model")
        print(f"Best Validation Accuracy: {best_state['acc']:.4f} | Params: {best_state['params']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run_optuna_study(trials=args.trials, epochs=args.epochs)
