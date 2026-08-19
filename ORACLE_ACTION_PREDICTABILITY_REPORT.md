# Oracle-action predictability — 2026-08-19

## Protokół

Zamrożony checkpoint utility-only Tiny ImageNet seed 123 generuje dla każdego
obrazu label-informed oracle action: `noop` albo jeden z 12 legalnych
transferów po cztery grupy. Mała głowa MLP otrzymuje wyłącznie 64-wymiarowe
cechy po stemie. Uczono ją przez 300 kroków na 512 przykładach treningowych i
oceniono na osobnych 512 obrazach walidacyjnych.

## Wyniki

| Split | Accuracy akcji | Majority baseline | Mean regret względem oracle |
|---|---:|---:|---:|
| Train | 52,34% | 43,16% | 0,018 NLL |
| Validation | 20,90% | 23,83% | 1,534 NLL |

## Wniosek

Cecha po stemie nie przewiduje przenoszalnie decyzji oracle'a. Model potrafi
zapamiętać część etykiet treningowych, ale nie przebija nawet klasy
większościowej na walidacji i traci praktycznie cały zysk oracle'a.

Nie integrujemy globalnego action head. Następny prototyp powinien podejmować
decyzje sekwencyjnie po obserwacji cech z kolejnych etapów ResNet, ograniczając
akcje do budżetu przyszłych bloków. Tylko taka policy ma dostęp do informacji,
która może rozróżniać obrazy wymagające dodatkowego obliczenia.
