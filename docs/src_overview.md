# Documentacao do codigo (src)

Este documento explica, de forma didatica, como o codigo em `src/` esta organizado e como os dados fluem entre os binarios.

## Fluxo geral (do audio ate a avaliacao)

1. `organize_dataset`: reorganiza os arquivos WAV descritos no `metadata.csv` para a estrutura `data/<classe>/<id>.wav` (copia ou link).
2. `preprocess`: le os WAVs, normaliza amplitude, divide em frames e grava em `processed/<classe>/<id>.frames`.
3. `extract_features`: le os frames, calcula MFCC + estatisticas espectrais por frame e agrega por arquivo (media e desvio).
4. `split_normalize`: separa treino/teste por locutor, normaliza (z-score) e gera `train.data`/`test.data` (formato FANN).
5. `train`: treina a MLP (FANN) e salva o modelo.
6. `evaluate`: aplica o modelo e produz metricas + matriz de confusao.

Extras:
- `cross_validate`: faz k-fold com separacao por locutor e busca de hiperparametros.
- `svm_baseline`: baseline com libsvm usando os mesmos dados normalizados.

## Modulos de suporte

- `csv.c/.h`: parser CSV simples com suporte a campos entre aspas e aspas escapadas.
- `util.c/.h`: utilitarios de filesystem (mkdir recursivo, existencia e copia de arquivos).
- `frame_io.c/.h`: define o formato binario `.frames` (cabecalho + frames float).
- `dsp.c/.h`: funcoes basicas de DSP (janela de Hamming e FFT radix-2).
- `mfcc.c/.h`: cria filtro mel e calcula MFCC (energia em mel -> log -> DCT).

## Detalhes importantes por binario

### organize_dataset.c
- Le `metadata.csv`, encontra `id`, `classe` e `filepath`.
- Cria `data/<classe>/` e copia (ou cria symlink) para `id.wav`.
- Tem modo `--dry-run` e log opcional.

### preprocess.c
- Usa libsndfile para ler WAVs e converte para mono (media dos canais).
- Normaliza pelo pico absoluto para reduzir variacao de ganho.
- Faz framing com `frame_ms`/`hop_ms`.
- Opcionalmente remove silencio: calcula RMS por frame e usa um limiar relativo ao maximo.

### extract_features.c
- Para cada `id`/`classe` no metadata, le o arquivo `.frames` correspondente.
- Para cada frame:
  - RMS e ZCR (tempo).
  - FFT + magnitude (freq).
  - Centroid e rolloff (freq).
  - MFCC (filtro mel + DCT).
- Agrega por arquivo: media e desvio padrao de cada feature.

### split_normalize.c
- Carrega `features.csv` e `metadata.csv`.
- Garante separacao por locutor para evitar leakage.
- Faz um split aproximando a proporcao de classes.
- Calcula media/desvio apenas no treino e normaliza treino/teste.
- Escreve `train.data`/`test.data` no formato FANN e `classes.txt`.

### train.c
- Le `train.data` e cria uma MLP (1 ou 2 camadas ocultas).
- Treina com RPROP e salva o modelo em `.net`.
- Pode registrar MSE por epoca.

### evaluate.c
- Le o modelo treinado, executa no `test.data` e calcula:
  - acuracia
  - precisao/recall macro
  - matriz de confusao

### cross_validate.c
- Agrupa amostras por locutor, distribui em k folds balanceados.
- Para cada combinacao de hiperparametros, treina/avalia e escreve media e desvio.

### svm_baseline.c
- Reaproveita `train.data`/`test.data` (FANN) e treina um SVM RBF.
- Gera metricas e matriz de confusao no mesmo formato do avaliador.

## Formato `.frames`

O arquivo `.frames` comeca com um `FrameHeader` (sample_rate, frame_len, hop_len, num_frames) e em seguida os frames em `float` armazenados de forma contigua.
