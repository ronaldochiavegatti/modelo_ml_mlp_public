# SVD Voice Pathology Classification Pipeline (C)

This repository implements a full, modular CLI pipeline in C for classifying vocal pathologies using the Saarbrucken Voice Database (SVD). The baseline model is an MLP (FANN), with evaluation metrics and a confusion matrix visualization.

## Scope and classes

See `docs/scope.md` for the class selection and inclusion rules. The four classes are:
- laringite
- disfonia_funcional
- disfonia_psicogenica
- edema_de_reinke

## Directory layout

```
.
├── data/                     # Organized dataset: data/<classe>/<id>.wav
├── processed/                # Preprocessed frames: processed/<classe>/<id>.frames
├── features.csv              # Aggregated features per file
├── train.data                # FANN training data
├── test.data                 # FANN test data
├── classes.txt               # Class order used for one-hot encoding
├── model.net                 # Trained FANN model
├── results.csv               # Metrics + confusion matrix
├── confusion.svg             # Confusion matrix plot
├── docs/
│   ├── scope.md
│   └── feature_engineering.md
└── src/
    └── *.c
```

## Dependencies

- C compiler (gcc/clang)
- libsndfile (audio I/O)
- FANN (MLP training)
- libsvm (optional SVM baseline)

Ubuntu/Debian example:
```
sudo apt install libsndfile1-dev libfann-dev libsvm-dev
```

## Build

```
make all
```

Binaries are built into `bin/`.

## Main flows (quick testing)

### 1) Pipeline Principal (Recomendado)

**Update:** Após a otimização de hiperparâmetros (cross-validation) e a inclusão de features espectrais adicionais, a melhor configuração local (seed=42) atingiu aproximadamente **36.9%** de acurácia. Em 5 partições diferentes (seeds 42/123/7/99/202), a média ficou em **33.1% ± 2.3 pp**. Os comandos abaixo refletem essa configuração (resultados variam com a partição; ver tabela de resultados).

Este pipeline utiliza o extrator de features enriquecido (com deltas, F0, etc.) e remoção de silêncio para obter a melhor performance.

```
make all
./bin/organize_dataset --metadata metadata.csv --output data --log logs/verify.log
./bin/preprocess --input data --output processed_silence --frame-ms 30 --hop-ms 10 --remove-silence --silence-threshold 0.1
./bin/extract_features --input processed_silence --metadata metadata.csv --output features.csv --n-mfcc 13 --n-mels 26 --rolloff 0.85
./bin/split_normalize --features features.csv --metadata metadata.csv --train train.data --test test.data --classes classes.txt --train-ratio 0.7 --seed 42 --scaler scaler.csv
./bin/train --train train.data --model model.net --hidden 32 --hidden2 8 --learning-rate 0.0005 --max-epochs 300 --log train.log --seed 42
./bin/evaluate --model model.net --test test.data --classes classes.txt --output results.csv
./bin/plot_confusion --input results.csv --output confusion.svg --title "Confusion Matrix (Best Tuned Features)"
```

Atalho: `./scripts/run_best_pipeline.sh` (aceita `SEED` e `TRAIN_SEED` como variáveis de ambiente).

### 2) Pipeline Alternativa (sem remoção de silêncio)

Útil para comparação e análise, mas geralmente resulta em menor acurácia.

```
make all
./bin/organize_dataset --metadata metadata.csv --output data --log logs/verify.log
./bin/preprocess --input data --output processed --frame-ms 30 --hop-ms 10
./bin/extract_features --input processed --metadata metadata.csv --output features_no_silence.csv --n-mfcc 13 --n-mels 26 --rolloff 0.85
./bin/split_normalize --features features_no_silence.csv --metadata metadata.csv --train train_no_silence.data --test test_no_silence.data --classes classes_no_silence.txt --train-ratio 0.7 --seed 42 --scaler scaler_no_silence.csv
./bin/train --train train_no_silence.data --model model_no_silence.net --hidden 256 --hidden2 0 --learning-rate 0.001 --max-epochs 600 --desired-error 0.001 --log train_no_silence.log
./bin/evaluate --model model_no_silence.net --test test_no_silence.data --classes classes_no_silence.txt --output results_no_silence.csv
./bin/plot_confusion --input results_no_silence.csv --output confusion_no_silence.svg --title "Confusion Matrix (No Silence Removal)"
```

### Visualizar os pesos treinados

O SVG do modelo mostra o layout das camadas, mas os valores reais estão no arquivo `.net`. Use `scripts/dump_fann_weights.py` para gerar uma planilha CSV por ligação entre camadas:

```
python scripts/dump_fann_weights.py --model model.net
```

Os arquivos resultantes vão para `weights/model/`.

### Cross-validation + tuning (k=5)

```
./bin/cross_validate --features features.csv --metadata metadata.csv --k 5 \
  --hidden 16,32,64,128 --hidden2 0 --learning-rate 0.01,0.001 --max-epochs 300 --seed 42 \
  --output cv_report.csv
```

### Baseline SVM

```
./bin/svm_baseline --train train.data --test test.data --classes classes.txt --output svm_results.csv --c 1.0 --gamma 0.0 --model svm_model.svm
```

### Reset do ambiente (limpeza total)

O comando abaixo apaga **todos** os arquivos gerados, restaurando o projeto ao seu estado original.

```
make clean
rm -rf data processed processed_silence logs weights
rm -f features*.csv train*.data test*.data classes*.txt scaler*.csv \
  model*.net model*.svg results*.csv confusion*.svg confusion*.png train*.log \
  svm_model*.svm svm_results*.csv cv_report*.csv cv.log
```

## Step-by-step usage

### 1) Define metadata (no code)

Create `metadata.csv` using the schema described in `docs/scope.md`.
Use `metadata.example.csv` as a template.

### 2) Organize dataset

```
./bin/organize_dataset --metadata metadata.csv --output data --log logs/verify.log
```

### 3) Preprocess audio

```
./bin/preprocess --input data --output processed_silence --frame-ms 30 --hop-ms 10 --remove-silence --silence-threshold 0.1
```

### 4) Extract features

```
./bin/extract_features --input processed_silence --metadata metadata.csv --output features.csv --n-mfcc 13 --n-mels 26 --rolloff 0.85
```

This executable was significantly enhanced to extract a rich set of features suitable for pathology detection.

**Aggregated features per file:**
- **Static Features (Mean, Std, Skew, Kurtosis):**
  - 13 MFCCs
  - RMS energy
  - Zero-Crossing Rate (ZCR)
  - Crest Factor
  - Spectral Centroid
  - Spectral Rolloff
  - Spectral Bandwidth
  - Spectral Flatness
  - Spectral Flux
- **Temporal Features (Mean, Std, Skew, Kurtosis):**
  - 13 Delta MFCCs (1st derivative)
  - 13 Delta-Delta MFCCs (2nd derivative)
- **Pitch and Voicing Features:**
  - F0 Mean & Std (from voiced frames)
  - Voicing Rate (ratio of voiced to total frames)
  - Harmonicity Mean & Std (from ACF peak)

For a detailed explanation of the feature engineering process, see `docs/feature_engineering.md`.

### 5) Split + normalize (z-score)

```
./bin/split_normalize --features features.csv --metadata metadata.csv --train train.data --test test.data --classes classes.txt --train-ratio 0.7 --seed 42 --scaler scaler.csv
```

### 6) Train MLP (FANN, melhor configuração atual)

```
./bin/train --train train.data --model model.net --hidden 32 --hidden2 8 --learning-rate 0.0005 --max-epochs 300 --desired-error 0.001 --log train.log --seed 42
```

### 7) Evaluate

```
./bin/evaluate --model model.net --test test.data --classes classes.txt --output results.csv
```

### 8) Confusion matrix plot

```
./bin/plot_confusion --input results.csv --output confusion.svg --title "SVD Confusion Matrix"
```

## Notes

- All steps are independent CLI executables; outputs from one step are inputs to the next.
- For consistent MFCCs, keep a fixed sample rate across the dataset (resample during conversion if needed).
- The baseline target accuracy is ~87% following common SVD literature practices.

## Results snapshot (local runs)

The numbers below are from local runs on this repo with the current feature set (including spectral flux). Most runs use seed 42; one row uses a different split (seed 123) to illustrate variance. Results may vary with data splits.

| Setup | Hidden | Hidden2 | LR | Epochs | Accuracy | Precision (macro) | Recall (macro) | Outputs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (advanced features) | 128 | 0 | 0.01 | 300 | 0.358127 | 0.350306 | 0.342581 | `model.net`, `results.csv` |
| Tuned (CV best, small grid) | 16 | 0 | 0.001 | 300 | 0.360882 | 0.344897 | 0.339349 | `model_tuned.net`, `results_tuned.csv` |
| Tuned (expanded grid, 600 epochs) | 128 | 0 | 0.005 | 600 | 0.358127 | 0.338561 | 0.340173 | `model_tuned2.net`, `results_tuned2.csv` |
| Tuned + early stop (DE=0.03) | 16 | 0 | 0.01 | 1000 | 0.325069 | 0.308688 | 0.307520 | `model_tuned3_early.net`, `results_tuned3_early.csv` |
| Tuned + hidden2 (DE=0.02) | 64 | 32 | 0.001 | 1000 | 0.330579 | 0.307281 | 0.294216 | `model_tuned_hidden2_de02.net`, `results_tuned_hidden2_de02.csv` |
| Tuned + hidden2 (DE=0.05) | 64 | 32 | 0.001 | 1000 | 0.338843 | 0.329631 | 0.330530 | `model_tuned_hidden2_de05.net`, `results_tuned_hidden2_de05.csv` |
| Tuned + hidden2 (small grid best) | 32 | 8 | 0.0005 | 300 | **0.369146** | **0.357196** | **0.352221** | `model_tuned_hidden2_small.net`, `results_tuned_hidden2_small.csv` |
| Tuned + hidden2 (small grid, split seed=123) | 32 | 8 | 0.0005 | 300 | 0.327869 | 0.312481 | 0.310660 | `model_tuned_hidden2_small_seed123.net`, `results_tuned_hidden2_small_seed123.csv` |
| Tuned + hidden2 (small grid, split seed=7) | 32 | 8 | 0.0005 | 300 | 0.311475 | 0.295377 | 0.292436 | `model_tuned_hidden2_small_seed7.net`, `results_tuned_hidden2_small_seed7.csv` |
| Tuned + hidden2 (small grid, split seed=99) | 32 | 8 | 0.0005 | 300 | 0.330601 | 0.321344 | 0.315779 | `model_tuned_hidden2_small_seed99.net`, `results_tuned_hidden2_small_seed99.csv` |
| Tuned + hidden2 (small grid, split seed=202) | 32 | 8 | 0.0005 | 300 | 0.313889 | 0.301230 | 0.291948 | `model_tuned_hidden2_small_seed202.net`, `results_tuned_hidden2_small_seed202.csv` |
| Tuned + hidden2 (small grid, mean ± std, seeds 42/123/7/99/202) | 32 | 8 | 0.0005 | 300 | 0.3306 ± 0.0231 | 0.3175 ± 0.0243 | 0.3126 ± 0.0246 | `results_tuned_hidden2_small*.csv` |

## Future work

Trabalhos futuros: uso de Transformer como extrator de embeddings.
# modelo_ml_mlp
