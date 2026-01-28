# Seed Stability Report

Configuração avaliada:
- Modelo: MLP (FANN)
- Hidden: 32
- Hidden2: 8
- Learning rate: 0.0005
- Max epochs: 300
- Desired error: 0.001
- Train seed (FANN): 42
- Split seed(s): 7, 21, 42, 77, 99, 123, 202, 303

| Seed | Accuracy | Precision (macro) | Recall (macro) |
| --- | --- | --- | --- |
| 7 | 0.322404 | 0.300038 | 0.301879 |
| 21 | 0.322404 | 0.309644 | 0.305641 |
| 42 | 0.374656 | 0.357394 | 0.351620 |
| 77 | 0.349727 | 0.328176 | 0.323562 |
| 99 | 0.303279 | 0.294016 | 0.297222 |
| 123 | 0.316940 | 0.317203 | 0.310284 |
| 202 | 0.336111 | 0.323821 | 0.323920 |
| 303 | 0.330623 | 0.312338 | 0.310924 |

Resumo (média ± desvio padrão):
- Accuracy: 0.332018 ± 0.022001
- Precision (macro): 0.317829 ± 0.019601
- Recall (macro): 0.315632 ± 0.017343
