# Static-depth vs adaptive-depth — Tiny ImageNet, seed 42

## Protokół

Oba modele używają ResNet-50 dla małych obrazów, batch size 64/GPU, AdamW i 20
epok. Static-depth kończy zawsze po stage 3; adaptive-depth ma auxiliary exits
i wybiera exit przez threshold confidence kalibrowany na osobnej połowie
walidacji.

| Model | Held-out / validation Top-1 | Cost fraction |
|---|---:|---:|
| static depth, stage 3 | 53,55% | 81,25% |
| adaptive-depth confidence | 54,96% | 84,75% |
| adaptive full-depth | 54,96% | 100,00% |

## Interpretacja

Dynamiczna policy daje +1,41 pp względem stałego stage 3 za 3,5 pp dodatkowego
średniego kosztu, a jednocześnie zachowuje accuracy full-depth. Jest to
pozytywny punkt Pareto dla seeda 42, ale nie dowód końcowy: static-depth musi
zostać powtórzony dla seedów 123 i 2026, zanim porównanie będzie raportowane
jako średnia ± SD.
