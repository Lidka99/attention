# Replikacja adaptive-depth — Tiny ImageNet, 20 epok

## Zamrożony protokół

Trzy seedy: 42, 123 i 2026. ResNet-50 z głowami po stage 2, 3 i 4; wspólna
strata `0.25/0.5/1.0`; 20 epok; target średniego kosztu 85%. Progi confidence
dobierano na pierwszej połowie walidacji i oceniano na drugiej połowie.

## Wyniki held-out

| Seed | Policy Top-1 | Full-depth Top-1 | Różnica | Cost |
|---:|---:|---:|---:|---:|
| 42 | 54,96% | 54,96% | 0,00 pp | 84,75% |
| 123 | 53,46% | 53,78% | −0,32 pp | 85,08% |
| 2026 | 54,12% | 54,02% | +0,10 pp | 84,95% |
| średnia ± SD | 54,18% ± 0,75 | 54,25% ± 0,62 | −0,07 pp ± 0,21 | 84,92% ± 0,17 |

## Oracle

W każdym seedzie label-informed oracle jest wyraźnie lepszy od końcowego
wyjścia i używa około 75–76% głębokości. To pozostawia lukę do ulepszania
policy, ale obecny confidence threshold już realizuje stabilny kompromis.

## Decyzja

**GO.** Pilot spełnia bramkę: średnia strata accuracy jest znacznie mniejsza
niż 0,5 pp, a koszt pozostaje nie większy niż 85%. Następny etap nie zwiększa
jeszcze danych ani liczby epok. Najpierw należy wdrożyć forward, który po
wyborze exit rzeczywiście nie wykonuje późniejszych bloków, i zmierzyć medianę
oraz P95 latency dla batch size 1 i praktycznego batcha.
