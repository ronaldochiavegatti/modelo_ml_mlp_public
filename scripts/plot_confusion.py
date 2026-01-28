#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_confusion(path: Path):
    labels = []
    rows = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] != "confusion":
                continue
            if row[1] == "actual/pred":
                labels = row[2:]
                continue
            rows.append([int(x) for x in row[2:]])
    if not labels or not rows:
        raise ValueError("No confusion matrix found in results.csv")
    return labels, np.array(rows, dtype=int)


def plot_confusion(labels, matrix, output_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to results.csv")
    parser.add_argument("--output", default="confusion.png", help="Output image path")
    parser.add_argument("--title", default="Confusion Matrix", help="Plot title")
    args = parser.parse_args()

    labels, matrix = read_confusion(Path(args.input))
    plot_confusion(labels, matrix, args.output, args.title)


if __name__ == "__main__":
    main()
