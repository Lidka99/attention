# Adaptive-depth / early-exit pilot — Tiny ImageNet, seed 123

## Model i trening

ResNet-50 otrzymał dodatkowe klasyfikatory po stage 2 i stage 3. Wszystkie
trzy głowy uczono wspólnie przez 20 epok na Tiny ImageNet, ze stratą ważoną
`0.25 * CE(stage2) + 0.5 * CE(stage3) + CE(stage4)`. Jest to pilot mechanizmu,
nie finalny wynik publikacyjny.

## Oracle potential po 20 epokach

| Metryka | Wynik |
|---|---:|
| stage 2 Top-1 | 35,67% |
| stage 3 Top-1 | 52,53% |
| stage 4 Top-1 | 54,22% |
| label-informed oracle Top-1 | 62,53% |
| oracle mean cost fraction | 76,05% |

Oracle może wybrać najwcześniejsze poprawne wyjście. Jest górną granicą, ale
wskazuje, że adaptive depth ma potencjał zarówno jakościowy, jak i kosztowy.

## Confidence policy z niezależną kalibracją

Walidację podzielono deterministycznie na pierwsze 5 000 obrazów do wyboru
progów i pozostałe 5 000 do oceny. Wybrano progi: stage 2 = 0,80, stage 3 =
0,55, z celem średniego kosztu nie większym niż 85% pełnej głębokości.

| Metryka na held-out 5k | Wynik |
|---|---:|
| confidence-policy Top-1 | 53,46% |
| full-depth Top-1 | 53,78% |
| różnica | −0,32 pp |
| mean cost fraction | 85,08% |
| exit po stage 2 / stage 3 | 10,58% / 47,84% |

W pilocie policy zachowuje accuracy w granicy 0,5 pp przy szacowanym spadku
kosztu o około 15%. W przeciwieństwie do channel gating, early exit może
faktycznie ominąć późniejsze bloki; wymaga to jednak osobnego pomiaru latency.

## Decyzja

**GO do replikacji, nie do skalowania.** Następnie należy:

1. zamrozić trening i wykonać seedy 42 oraz 2026;
2. użyć osobnego calibration splitu dla każdego seeda;
3. raportować accuracy–cost Pareto, ECE/risk-coverage i realną medianę/P95
   latency dla implementacji, która faktycznie przerywa wykonanie;
4. porównać z full-depth oraz static-depth, zanim rozważymy Tiny ImageNet 100
   epok lub ImageNet-100.
