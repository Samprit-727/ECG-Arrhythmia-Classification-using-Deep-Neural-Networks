"""Train intermediate-level ECG classifiers on the MIT-BIH heartbeat dataset.

Models: SVM, Random Forest, Decision Tree, ANN, and 1D CNN.
The test set is never resampled; only the training portion is balanced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D


CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unclassifiable"]


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a headerless MIT-BIH CSV: 187 signal values followed by a label."""
    frame = pd.read_csv(path, header=None)
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64)
    return x, y


def balance_training_data(x: np.ndarray, y: np.ndarray, samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Upsample/downsample every training class to the requested size."""
    parts = []
    for class_id in range(len(CLASS_NAMES)):
        indices = np.flatnonzero(y == class_id)
        chosen = resample(
            indices,
            replace=len(indices) < samples_per_class,
            n_samples=samples_per_class,
            random_state=42 + class_id,
        )
        parts.append(chosen)
    indices = np.concatenate(parts)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    return x[indices], y[indices]


def build_ann(input_length: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(input_length,)),
        Dense(256, activation="relu"),
        Dropout(0.30),
        Dense(128, activation="relu"),
        Dropout(0.20),
        Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(input_length: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(input_length, 1)),
        Conv1D(32, 5, activation="relu", padding="same"),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.30),
        Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_evaluation(name: str, y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> dict[str, float | str]:
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    (output_dir / f"{name}_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)), normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"{name}: normalized confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=160)
    plt.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/heartbeat/mitbih_train.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset hb/mitbih_test.csv"))
    parser.add_argument("--samples-per-class", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)

    x_all, y_all = load_dataset(args.train)
    x_test, y_test = load_dataset(args.test)
    x_train_raw, x_valid, y_train_raw, y_valid = train_test_split(
        x_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    x_train, y_train = balance_training_data(x_train_raw, y_train_raw, args.samples_per_class)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    joblib.dump(scaler, args.output / "scaler.joblib")

    results = []
    classical_models = {
        "svm": SVC(kernel="rbf", C=2.0, gamma="scale"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(max_depth=20, random_state=42),
    }
    for name, model in classical_models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        results.append(save_evaluation(name, y_test, model.predict(x_test), args.output))
        joblib.dump(model, args.output / f"{name}_model.joblib")

    callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ann = build_ann(x_train.shape[1])
    print("Training ANN...")
    ann.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=args.epochs, batch_size=128, callbacks=[callback])
    results.append(save_evaluation("ann", y_test, ann.predict(x_test, verbose=0).argmax(axis=1), args.output))
    ann.save(args.output / "ann_model.keras")

    cnn = build_cnn(x_train.shape[1])
    print("Training CNN...")
    cnn.fit(x_train[..., np.newaxis], y_train, validation_data=(x_valid[..., np.newaxis], y_valid), epochs=args.epochs, batch_size=128, callbacks=[callback])
    results.append(save_evaluation("cnn", y_test, cnn.predict(x_test[..., np.newaxis], verbose=0).argmax(axis=1), args.output))
    cnn.save(args.output / "cnn_model.keras")

    table = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    table.to_csv(args.output / "model_comparison.csv", index=False)
    print("\nTest-set comparison (the test data was not balanced):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
