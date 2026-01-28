# Pipeline: execucao e resultados

Este guia mostra como executar a pipeline completa e coletar os resultados do modelo.

## Pre-requisitos

- Dependencias: ver `README.md` (libsndfile, fann, libsvm opcional).
- Rode os comandos a partir da raiz do repositorio.

## Passo a passo

1) Criar o `metadata.csv`

- Use `metadata.example.csv` como base.
- Consulte `docs/scope.md` para regras de classe e filtros.

2) Compilar os binarios

```bash
make all
```

3) Organizar o dataset (copia ou link)

```bash
./bin/organize_dataset --metadata metadata.csv --output data --log logs/verify.log
```

4) Preprocessar audio (frames)

```bash
./bin/preprocess --input data --output processed --frame-ms 30 --hop-ms 10
```

Opcional: remocao de silencio

```bash
./bin/preprocess --input data --output processed_silence --frame-ms 30 --hop-ms 10 --remove-silence --silence-threshold 0.1
```

5) Extrair features

```bash
./bin/extract_features --input processed --metadata metadata.csv --output features.csv --n-mfcc 13 --n-mels 26 --rolloff 0.85
```

Se usou `processed_silence`, ajuste `--input` e o arquivo de saida (ex: `features_silence.csv`).

6) Split + normalizacao

```bash
./bin/split_normalize --features features.csv --metadata metadata.csv --train train.data --test test.data --classes classes.txt --train-ratio 0.7 --seed 42 --scaler scaler.csv
```

7) Treinar a MLP (FANN)

```bash
./bin/train --train train.data --model model.net --hidden 32 --hidden2 0 --learning-rate 0.01 --max-epochs 500 --desired-error 0.001 --log train.log
```

8) Avaliar o modelo

```bash
./bin/evaluate --model model.net --test test.data --classes classes.txt --output results.csv
```

9) Plotar matriz de confusao

```bash
./bin/plot_confusion --input results.csv --output confusion.svg --title "SVD Confusion Matrix"
```

## Resultados gerados

- `results.csv`: metricas (accuracy, precision/recall macro e por classe) e matriz de confusao.
- `confusion.svg`: visualizacao da matriz de confusao.
- `model.net`: modelo treinado.
- `train.log`: historico do erro por epoca (se habilitado).
- `classes.txt`: ordem de classes usada no one-hot.
- `scaler.csv`: medias e desvios usados na normalizacao.

## Resultados recentes (silence)

- Modelo: `model_silence_best.net` (hidden1=128, hidden2=0, lr=0.001, 400 epocas).
- Avaliacao: `results_silence.csv` e `confusion_silence.svg`.
- Metricas: accuracy 0.391185, precision macro 0.374672, recall macro 0.367180.

## Extras (opcional)

- Cross-validation e tuning:

```bash
./bin/cross_validate --features features.csv --metadata metadata.csv --k 5 --hidden 32,64,128 --hidden2 0,32 --learning-rate 0.01,0.001 --max-epochs 300 --seed 42 --output cv_report.csv
```

- Cross-validation focada (silence, lr=0.001, 600 epocas):

```bash
./bin/preprocess --input data --output processed_silence --frame-ms 30 --hop-ms 10 --remove-silence --silence-threshold 0.1
./bin/extract_features --input processed_silence --metadata metadata.csv --output features_silence.csv --n-mfcc 13 --n-mels 26 --rolloff 0.85
./bin/cross_validate --features features_silence.csv --metadata metadata.csv --k 5 --hidden 64,128,192,256 --hidden2 0,32 --learning-rate 0.001 --max-epochs 600 --seed 42 --output cv_report_silence_focus.csv
```

Melhor linha observada: `256,0,0.001000,0.223212,0.184470`. Observacao: a grade maior 7x3 com 600 epocas apresentou timeout duas vezes; reduzi o tamanho do lote para evitar timeouts.

- Baseline SVM:

```bash
./bin/svm_baseline --train train.data --test test.data --classes classes.txt --output svm_results.csv --c 1.0 --gamma 0.0 --model svm_model.svm
```

## Troubleshooting (erros comuns)

- `fatal error: sndfile.h: No such file or directory`: instale `libsndfile1-dev` e rode `make clean && make all`.
- `fatal error: fann.h: No such file or directory`: instale `libfann-dev` e rode `make clean && make all`.
- `ld: cannot find -lsvm`: instale `libsvm-dev` ou pule o passo do SVM.
- `plot_confusion: No confusion matrix found`: confirme se o `results.csv` tem as linhas `confusion,*` e se o caminho de entrada esta correto.
- `Missing required columns: id, classe, filepath`: confira o header do `metadata.csv` e use separador `,`.
- `Empty metadata file` ou `No samples loaded`: verifique se o `metadata.csv` e o `features.csv` tem linhas de dados.
- `Missing frames: ...`: rode `preprocess` e confirme `processed/<classe>/<id>.frames` com nomes iguais aos do `metadata.csv`.
- `Class count mismatch` ou `Train/test dimension mismatch`: gere novamente `train.data`, `test.data` e `classes.txt` com `split_normalize`.
- `Processed ... (0 frames)`: reduza `--silence-threshold` ou desative `--remove-silence`.
- `Failed to open ...`: confirme caminhos e permissao de leitura/escrita.

## Reset do ambiente (arquivos gerados)

Comando unico para limpar builds e artefatos da pipeline:

```bash
make clean
rm -rf data processed processed_silence logs
rm -f features.csv features_silence.csv train.data test.data classes.txt classes_silence.txt \
  model.net model_silence.net model_silence_best.net results.csv results_silence.csv \
  confusion.svg confusion_silence.svg train.log train_silence.log train_silence_best.log \
  scaler.csv scaler_silence.csv svm_model.svm svm_results.csv cv_report*.csv
```

Ajuste a lista se quiser preservar algum resultado.
