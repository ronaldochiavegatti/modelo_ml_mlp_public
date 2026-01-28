# Documentação de Engenharia de Features

Este documento detalha as melhorias aplicadas ao processo de extração de features (`bin/extract_features`) para aumentar a capacidade do modelo de classificar patologias vocais.

## 1. Melhoria da Agregação Estatística

Além da média (`mean`) e do desvio padrão (`std`), foram adicionadas duas novas medidas estatísticas para agregar os valores das features por frame:

-   **Assimetria (Skewness)**: Mede a assimetria da distribuição dos valores. Pode indicar se há picos ou vales incomuns na variação de uma feature.
-   **Curtose (Kurtosis)**: Mede o "achatamento" ou a "pontudez" da distribuição. Ajuda a identificar a presença de outliers ou variações extremas.

**Justificativa**: Fornece uma descrição estatística muito mais completa da distribuição de cada feature ao longo do tempo, capturando nuances que a média e o desvio padrão sozinhos não conseguem.

## 2. Features Dinâmicas: Deltas e Delta-Deltas

Foram adicionadas as derivadas temporais de 1ª e 2ª ordem para todos os 13 coeficientes MFCC.

-   **Delta MFCCs**: `delta[t] = mfcc[t+1] - mfcc[t-1]`
-   **Delta-Delta MFCCs**: `delta2[t] = delta[t+1] - delta[t-1]`

**Justificativa**: Patologias vocais frequentemente alteram a dinâmica da fala, ou seja, a forma como o espectro sonoro varia no tempo. Os MFCCs estáticos perdem essa informação. Os deltas capturam essa dinâmica, fornecendo ao modelo pistas cruciais sobre a instabilidade ou a lentidão das transições espectrais, o que é um forte diferenciador entre patologias.

## 3. Features de Pitch e Vozeamento

Foi implementado um algoritmo de **Autocorrelação (ACF)** para estimar a frequência fundamental (F0 ou pitch) e características de vozeamento em cada frame.

**Features extraídas por arquivo:**

-   `f0_mean` / `f0_std`: A média e o desvio padrão do F0, calculados apenas nos frames classificados como "vozeados". Isso mede o tom médio e a instabilidade do pitch (jitter).
-   `voicing_rate`: A proporção de frames vozeados em relação ao total de frames no áudio. Indica o quão contínua é a fonação.
-   `harmonicity_mean` / `harmonicity_std`: A média e o desvio padrão do pico da autocorrelação normalizada. É um proxy para a "armonicidade" ou a força do sinal periódico, ajudando a medir a presença de ruído ou soprosidade na voz (shimmer).

**Justificativa**: O F0 é uma das características mais importantes da voz humana. Sua estabilidade, valor médio e a presença de vozeamento são diretamente afetados por inúmeras patologias que alteram a vibração das cordas vocais. Essas features são extremamente poderosas para este tipo de classificação.

## 4. Features Espectrais Complementares

Foram adicionadas três medidas espectrais que complementam centroid e rolloff já utilizados:

-   **Spectral Bandwidth**: Mede a dispersão da energia ao redor do centroide espectral. Indica se o espectro é mais concentrado ou mais espalhado.
-   **Spectral Flatness**: Razão entre média geométrica e média aritmética do espectro; indica o quão "tonal" (baixo) ou "ruidoso" (alto) é o sinal.
-   **Spectral Flux**: Mede a variação do espectro entre frames consecutivos, capturando mudanças rápidas na distribuição de energia.

**Justificativa**: Patologias vocais frequentemente aumentam ruído e instabilidade espectral. Bandwidth e flatness ajudam a quantificar espalhamento e ruidez, enquanto flux captura mudanças rápidas de timbre que não são visíveis em estatísticas estáticas isoladas.

## 5. Fórmulas e parâmetros (implementação em C)

Esta seção descreve o cálculo exato implementado no `bin/extract_features` e parâmetros relevantes. Todas as features por frame são agregadas por **média, desvio padrão, assimetria e curtose** (mean/std/skew/kurt).

### 5.1 Deltas e delta-deltas (MFCC)

Para cada coeficiente MFCC `c_t` (em frames `t = 0..T-1`):

-   `delta_0 = c_1 - c_0`
-   `delta_t = c_{t+1} - c_{t-1}` (para `1 <= t <= T-2`)
-   `delta_{T-1} = c_{T-1} - c_{T-2}`

Os **delta-deltas** são calculados aplicando a mesma regra sobre a sequência de deltas.

### 5.2 Pitch/Voicing/Harmonicity (ACF)

Implementação em `src/dsp.c` (autocorrelação normalizada):

-   **Faixa de F0**: `min_f0 = 75 Hz`, `max_f0 = 500 Hz`
-   **Lags**: `min_lag = sample_rate / max_f0`, `max_lag = sample_rate / min_f0` (limitado a `frame_len - 1`)
-   **ACF normalizada**:
    -   `r(lag) = sum_{i=0}^{N-lag-1} x[i] * x[i+lag] / sum_{i=0}^{N-1} x[i]^2`
-   **Harmonicity**: valor máximo de `r(lag)` (`best_lag_val`)
-   **Voicing**: frame é vozeado se `best_lag_val > 0.85`
-   **F0**: `f0 = sample_rate / best_lag` (quando vozeado); caso contrário `0.0`

Agregação:
-   `f0_mean` / `f0_std`: apenas nos frames vozeados
-   `voicing_rate`: `n_voiced / n_frames`
-   `harmonicity_mean` / `harmonicity_std`: todos os frames

### 5.3 Espectrais (Centroid/Rolloff/Bandwidth/Flatness)

Pré-processamento por frame:
-   **Janela de Hamming**: `w[n] = 0.54 - 0.46 * cos(2πn/(N-1))`
-   **FFT**: `fft_size = next_pow2(frame_len)`
-   **Bins**: `f_k = k * sample_rate / fft_size`, para `k = 0..(fft_size/2)`

Usando a magnitude `mag[k]`:

-   **Spectral Centroid**:
    -   `centroid = sum_k (mag[k] * f_k) / sum_k mag[k]`
-   **Spectral Rolloff**:
    -   menor `f_k` tal que `sum_{i=0}^k mag[i] >= rolloff_pct * sum_k mag[k]`
    -   `rolloff_pct` vem de `--rolloff` (padrão `0.85`)
-   **Spectral Bandwidth**:
    -   `bandwidth = sqrt( sum_k ((f_k - centroid)^2 * mag[k]) / sum_k mag[k] )`
-   **Spectral Flatness**:
    -   `flatness = geo_mean / arith_mean`
    -   `geo_mean = exp( (1/N) * sum_k log(mag[k] + 1e-9) )`
    -   `arith_mean = sum_k mag[k] / N`

### 5.4 Spectral Flux

Definição usada no `bin/extract_features`:

-   `flux_t = sqrt( (1/N) * sum_k (mag_t[k] - mag_{t-1}[k])^2 )`

No primeiro frame, define-se `flux_0 = 0`. O fluxo é então agregado por mean/std/skew/kurt junto às demais features estáticas.
