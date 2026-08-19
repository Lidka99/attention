# Oracle audit — Tiny ImageNet, 2026-08-19

## Cel

Sprawdzić, czy przy aktualnym modelu istnieje użyteczny sygnał do lokalnego
przenoszenia budżetu między etapami, zanim zmienimy target albo policy.

Oracle korzysta z etykiety tylko podczas audytu: dla każdego obrazu wybiera
najniższy NLL spośród aktualnej alokacji i wszystkich prawidłowych,
kierunkowych transferów czterech grup pomiędzy dwoma etapami. Nie jest to
procedura inferencyjna ani wynik do porównania wydajności.

## Protokół

- dane: pierwsze 512 obrazów walidacyjnych Tiny ImageNet;
- budżet: 40/64 grup, floor 4, transfer 4 grup;
- kandydaci: bieżąca alokacja oraz wszystkie legalne transfery donor → recipient;
- checkpointy: końcowe po 100 epokach, seed 123.

## Wyniki

| Checkpoint | Baseline Top-1 | Oracle Top-1 | Zysk | Redukcja NLL | Zmienione obrazy |
|---|---:|---:|---:|---:|---:|
| value-of-compute | 60,16% | 71,09% | +10,94 pp | 1,533 | 81,05% |
| utility-only | 60,74% | 72,66% | +11,91 pp | 1,522 | 76,17% |

## Wniosek

Lokalna redystrybucja budżetu ma wyraźny potencjał. Porażka value-of-compute
nie wynika z braku sygnału w przestrzeni akcji, lecz z tego, że policy nie
przewiduje wyboru oracle'a. Następny krok: diagnostyka uczenia policy na
zamrożonych targetach (overfit), przed zmianą semantyki targetu.
