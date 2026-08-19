# Policy diagnostics — 2026-08-19

## Fixed-batch overfit

Checkpoint value-of-compute Tiny ImageNet, seed 123. Zamrożono 64 obrazy
walidacyjne i ich aktualne kontrfaktyczne targety, a następnie optymalizowano
wyłącznie `model.policy` przez 200 kroków AdamW (`lr=0.01`).

| Metryka | Początek | Koniec |
|---|---:|---:|
| stage-value loss | 1,388 | 1,053 |
| Top-1 agreement z targetem | 40,6% | 76,6% |

Głowa potrafi zapamiętać zamrożony target. Rozbieżność z pełnym treningiem
(agreement 12--13%) wskazuje na zbyt rzadki albo niestacjonarny sygnał, a nie
na brak gradientu lub pojemności modelu.

## Następny pilot

Dziesięć epok bez testu, przy niezmienionej semantyce targetu i wadze straty.
Jedyną zmianą jest `counterfactual_every: 1` zamiast 32. Celem jest agreement
większy od losowych 25%, nie wynik klasyfikacyjny.
