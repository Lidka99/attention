# Wyniki eksperymentów — attentionv3

Ten plik jest bieżącym, audytowalnym dziennikiem wyników. CIFAR-100 służy tu
wyłącznie do walidacji mechanizmu i protokołu przed Tiny ImageNet/ImageNet;
nie jest podstawą twierdzeń o wyniku na ImageNet.

## Protokół CIFAR-100

- model: ResNet-50 z UCLA channel attention po czterech etapach;
- obrazy: 32×32, stem ResNet dostosowany do CIFAR;
- sprzęt: 4 GPU przez DDP, batch size 128 na proces;
- harmonogram: 2 epoki warm-up, 6 sparsification, 2 fine-tuning;
- metryka jakości: Top-1 na 10 000 obrazów testowych;
- metryka kosztu: średni `keep_ratio` grup kanałów;
- UWAGA: `keep_ratio` nie jest rzeczywistym speedupem obecnej implementacji.

## Warianty

| Wariant | Znaczenie | Konfiguracja |
|---|---|---|
| UCLA | utility + uncertainty + adaptacyjny dodatkowy budżet | `cifar100_ucla_matched_0625.yaml` |
| utility-only | dynamiczna selekcja bez uncertainty | `cifar100_utility_only.yaml` |
| static | jedna uczona maska dla wszystkich obrazów | `cifar100_static.yaml` |
| full ResNet-50 | brak bramek, punkt odniesienia jakości | `cifar100_resnet50_baseline.yaml` |

Po wstępnym runie UCLA przy keep-ratio 74,2% budżet UCLA został dopasowany:
`budget: 0.50`, `max_budget: 0.75`, `adaptive_extra: 0.30`. Docelowo daje to
około 62,5% zachowanych grup, czyli porównywalny koszt z kontrolami.

## Wyniki ukończone

| Seed | Wariant | Top-1 | Keep-ratio | Interpretacja |
|---:|---|---:|---:|---|
| 42 | UCLA (pierwotny budżet) | 52,77% | 74,22% | Nieporównywalny kosztowo — tylko diagnostyka. |
| 42 | UCLA (dopasowany) | 52,79% | 62,70% | 0,37 pp poniżej static, 1,92 pp powyżej utility-only. |
| 42 | utility-only | 50,87% | 62,50% | Kontrola dynamicznej utility. |
| 42 | static | 53,16% | 62,50% | Najlepszy wynik dla seeda 42. |
| 123 | UCLA (dopasowany) | 53,25% | 62,60% | Najlepszy wariant dla seeda 123. |
| 123 | utility-only | 51,09% | 62,50% | Poniżej UCLA o 2,16 pp. |
| 123 | static | 52,05% | 62,50% | Poniżej UCLA o 1,20 pp. |
| 2026 | UCLA (dopasowany) | 47,02% | 62,04% | Znacząco słabszy seed; wymaga porównania z kontrolami. |
| 2026 | utility-only | 52,56% | 62,50% | Najlepszy wariant dla seeda 2026. |
| 2026 | static | 50,18% | 62,50% | Poniżej utility-only dla tego seeda. |
| 42 | full ResNet-50 | 52,73% | 100,00% | Pełny model bez bramek; punkt odniesienia jakości. |

## Podsumowanie trzech seedów

Wartości to średnia ± odchylenie standardowe z seedów 42, 123 i 2026.

| Wariant | Top-1 | Keep-ratio |
|---|---:|---:|
| UCLA (dopasowany) | 51,02% ± 3,47 pp | 62,45% ± 0,36 pp |
| utility-only | 51,51% ± 0,92 pp | 62,50% ± 0,00 pp |
| static | **51,80% ± 1,51 pp** | 62,50% ± 0,00 pp |

## Wnioski po trzech seedach

1. Przy prawie identycznym keep-ratio UCLA **nie wykazał przewagi średniej**:
   jest 0,49 pp poniżej utility-only i 0,78 pp poniżej static.
2. UCLA ma znacznie większą zmienność między seedami; słaby wynik 47,02% dla
   seeda 2026 determinuje średnią. To jest sygnał problemu stabilności, nie
   dowód przewagi uncertainty-aware controller.
3. Hipoteza, że uncertainty poprawia selekcję względem utility-only, pozostaje
   niepotwierdzona na tym protokole. Nie należy przechodzić jeszcze do
   Tiny ImageNet ani ImageNet z obecną konfiguracją jako „zwycięską”.
4. Następny etap powinien być diagnostyczny: baseline pełnego ResNet-50,
   analiza ECE/NLL/Brier i korelacji uncertainty–błąd, kontrola inicjalizacji
   oraz przegląd funkcji celu UCLA. Dopiero po usunięciu niestabilności warto
   ponowić zestaw seedów.

## Baseline pełnego modelu

Full ResNet-50 bez bramek uzyskał 52,73% Top-1 dla seeda 42. Nie jest to
bezpośrednio porównywalne kosztowo z variantami 62,5% keep-ratio, ale ustanawia
górny punkt jakości dla tej krótkiej, dziesięcioepokowej procedury.

## Diagnostyka uncertainty — UCLA, seed 42

Raport zapisany w `results/cifar100/seed42/ucla_matched_0625/analysis/`:

| Metryka | Wynik |
|---|---:|
| Top-1 | 52,79% |
| NLL | 1,8281 |
| ECE | 11,63% |
| Brier wieloklasowy | 0,6248 |
| Brier uncertainty względem błędu | 0,2486 |
| Korelacja Pearsona uncertainty–błąd | 0,142 |
| Średnia uncertainty: trafienia / błędy | 0,4445 / 0,4474 |

Uncertainty jest tylko minimalnie wyższa dla błędów niż dla trafień. W obecnej
wersji kontroler słabo rozróżnia ryzykowne decyzje; przed kolejnym tuningiem
trzeba przeanalizować funkcję celu i sposób definiowania targetu błędu.
