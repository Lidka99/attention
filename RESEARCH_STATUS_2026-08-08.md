# Status badań — 2026-08-08

## Cel

Badamy attention jako dynamiczny mechanizm kompresji obliczeń CNN: dla każdego
obrazu aktywujemy dokładnie 40 z 64 grup kanałów (62,5%) i rozdzielamy ten
budżet pomiędzy cztery etapy ResNet-50.

Nowa metoda `value-of-compute` uczy policy kontrfaktycznie: przewiduje, w
którym etapie ograniczenie grup kanałów najbardziej podnosi stratę.

## Co zostało odrzucone

Poprzedni global UCLA, oparty o ogólną uncertainty błędu, przegrał z
utility-only na pełnym CIFAR-100 seed 42: 69,82% vs 71,58%. Raport:
`FAILURE_REPORT_GLOBAL_UCLA_SEED42.md`.

## Co poprawiliśmy

- wydzielony split train/validation/test dla nowych pilotów CIFAR-100;
- kontrfaktyczny target stage value-of-compute;
- floor 4 grupy na etap;
- soft quota z exact rounding do 40 grup, zamiast greedy `[4,16,16,4]`;
- kontrola utility-only przy identycznym koszcie;
- hybrydowy Muon/AdamW jako osobna ablacją optymalizatora.

## Wyniki pilotów

| Zbiór / protokół | value-of-compute | utility-only | Wniosek |
|---|---:|---:|---|
| CIFAR-100, 20 epok, floor (wcześniejszy alokator) | 58,15% test | 58,80% test | Value nie wygrywa; wynik diagnostyczny. |
| CIFAR-100, 50 epok, soft quota, Muon | 66,44% wal. | — | Soft quota stabilizuje budżet. |
| CIFAR-100, 50 epok, soft quota, AdamW | 66,34% wal. | — | Muon ma słaby, jednoseedowy sygnał przewagi. |
| Tiny ImageNet, 20 epok, soft quota + Muon | **58,20% wal.** | **56,35% wal.** | Pierwszy dodatni sygnał: +1,85 pp przy identycznym budżecie. |

## Co ten wynik znaczy, a czego nie znaczy

Tiny ImageNet daje pierwszy konkretny powód, aby kontynuować: value-of-compute
wygrywa z utility-only o 1,85 pp przy identycznym budget i optimizerze.

Nie jest to jeszcze wynik publikacyjny, ponieważ mamy tylko jeden seed i 20
epok. Nie wolno twierdzić, że metoda jest już lepsza ogólnie ani raportować
realnego speedupu — obecne bramy nie omijają fizycznie konwolucji.

## Następne decyzje

1. Zamrozić aktualny soft quota, target, floor i Muon.
2. Powtórzyć Tiny ImageNet value i utility-only dla seeda 123, 20 epok.
3. Jeśli dodatnia różnica utrzyma się: uruchomić dłuższy protokół (50 epok)
   dla obu wariantów i dwóch seedów.
4. Dopiero potem wykonać końcowy audyt, trzeci seed i ewentualnie ConvNeXt V2.
5. Jeśli seed 123 nie potwierdzi efektu: potraktować Tiny wynik jako hipotezę,
   a nie dowód, i poprawić agreement value–counterfactual.

## Artefakty

- Tiny value: `results/tinyimagenet/value_budget_muon/seed42/`;
- Tiny utility control: `results/tinyimagenet/utility_floor_muon/seed42/`;
- stan i mapa kodu: `VALUE_BUDGET_RESEARCH_LOG.md`;
- wcześniejsza analiza porażki UCLA: `FAILURE_REPORT_GLOBAL_UCLA_SEED42.md`.
