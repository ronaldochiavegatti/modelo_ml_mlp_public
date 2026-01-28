#!/usr/bin/env bash
set -euo pipefail

# Best-known pipeline configuration (local runs).
# Usage:
#   SEED=42 TRAIN_SEED=42 ./scripts/run_best_pipeline.sh

SEED="${SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-42}"

make all
mkdir -p logs

./bin/organize_dataset --metadata metadata.csv --output data --log logs/verify.log
./bin/preprocess --input data --output processed_silence --frame-ms 30 --hop-ms 10 --remove-silence --silence-threshold 0.1
./bin/extract_features --input processed_silence --metadata metadata.csv --output features.csv --n-mfcc 13 --n-mels 26 --rolloff 0.85
./bin/split_normalize --features features.csv --metadata metadata.csv --train train.data --test test.data --classes classes.txt --train-ratio 0.7 --seed "${SEED}" --scaler scaler.csv

./bin/train --train train.data --model model.net --hidden 32 --hidden2 8 --learning-rate 0.0005 --max-epochs 300 --desired-error 0.001 --log train.log --seed "${TRAIN_SEED}"
./bin/evaluate --model model.net --test test.data --classes classes.txt --output results.csv
./bin/plot_confusion --input results.csv --output confusion.svg --title "Confusion Matrix (Best Tuned Features)"
