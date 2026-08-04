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

## Zamrożony protokół global attention — CIFAR-100, seed 42 (200 epok)

To jest osobny, pełny protokół: 20 epok warm-up, 120 sparsification i 60
fine-tuningu, 4 GPU DDP, globalnie dokładnie 40 z 64 grup (62,5%) dla każdego
obrazu. Wyniki dotyczą checkpointów końcowych (`latest.pt`) po z góry
ustalonych 200 epokach, nie checkpointów wybranych po najlepszym wyniku testu.

| Wariant | Top-1 | Bootstrap 95% CI | Keep-ratio |
|---|---:|---:|---:|
| global UCLA | 69,82% | [68,95%; 70,70%] | 62,50% |
| global utility-only | **71,58%** | [70,66%; 72,50%] | 62,50% |

Sparowana różnica `global UCLA − utility-only` wynosi **−1,76 pp**, z 95% CI
**[−2,58; −0,93] pp**. Utility-only ma poprawną predykcję tam, gdzie UCLA jej
nie ma, dla 1006 obrazów; odwrotna sytuacja występuje dla 830 obrazów.

### Diagnostyka niepewności global UCLA

| Metryka | Wynik |
|---|---:|
| Top-1 / Top-5 | 69,82% / 90,22% |
| NLL / ECE | 2,4737 / 22,50% |
| Brier wieloklasowy | 0,5075 |
| Korelacja Pearsona uncertainty–błąd | **0,054** |
| Średnia uncertainty: trafienia / błędy | 0,00260 / 0,00280 |

### Decyzja go/no-go

**Nie przechodzi.** UCLA jest gorszy od utility-only o więcej niż dozwolone
0,5 pp, a korelacja uncertainty–błąd (0,054) jest znacznie poniżej progu 0,25.
Nie uruchamiać seedów 123/2026 ani Tiny ImageNet dla tej wersji mechanizmu.
Następny eksperyment musi najpierw przeprojektować uczenie i kalibrację
uncertainty, po czym przejść krótki pilot na CIFAR-100 przed ponownym pełnym
protokołem.
