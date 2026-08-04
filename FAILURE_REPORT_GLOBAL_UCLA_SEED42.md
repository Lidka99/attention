# Sprawozdanie z niepowodzenia: global UCLA vs utility-only, CIFAR-100 seed 42

Data: 2026-08-04  
Status: pełny eksperyment kontrolny zakończony; obecna wersja global UCLA nie przechodzi kryterium go/no-go.

## Cel i warunki porównania

Badaliśmy hipotezę, że predykowana niepewność może, przy dokładnie tym samym globalnym budżecie, rozdzielać obliczenia między cztery etapy ResNet-50 lepiej niż sama utility.

- CIFAR-100, 10 000 obrazów testowych;
- 4 GPU DDP, batch 128/GPU;
- 200 epok: 20 warm-up, 120 sparsification, 60 fine-tuningu;
- dokładnie 40 z 64 grup kanałów (62,5%) dla każdego obrazu;
- raportowany checkpoint po ustalonych 200 epokach (`latest.pt`), nie checkpoint wybrany po najlepszym teście.

`global UCLA` przydziela grupy między etapami według średniej predykowanej uncertainty etapu. `global utility-only` używa tej samej architektury i reguły twardego budżetu, lecz ranking etapów opiera na utility i nie uczy uncertainty.

## Wynik

| Wariant | Top-1 | Bootstrap 95% CI | Keep-ratio |
|---|---:|---:|---:|
| global UCLA | 69,82% | [68,95%; 70,70%] | 62,50% |
| global utility-only | **71,58%** | [70,66%; 72,50%] | 62,50% |

Sparowana różnica `UCLA − utility-only` wynosi **−1,76 pp**, z 95% CI **[−2,58; −0,93] pp**. Przedział nie obejmuje zera. Utility-only jest poprawny tam, gdzie UCLA się myli, dla 1006 obrazów; UCLA jest poprawny przy błędzie utility-only dla 830 obrazów.

## Co na pewno działało

- Oba runy ukończyły 200 epok na czterech GPU.
- Globalny koszt był identyczny: keep-ratio dokładnie 62,5%.
- Historia, checkpointy końcowe i audyt sparowany zostały zapisane.
- UCLA nauczył się klasyfikacji (69,82%), więc nie jest to awaria infrastruktury ani całego ResNeta.

Problem dotyczy wartości dodanej uncertainty i sposobu, w jaki steruje ona budżetem.

## Diagnostyka uncertainty UCLA

| Metryka | Wynik |
|---|---:|
| Top-1 / Top-5 | 69,82% / 90,22% |
| NLL | 2,4737 |
| ECE klasyfikatora | 22,50% |
| Brier wieloklasowy | 0,5075 |
| Korelacja Pearsona uncertainty–błąd | **0,054** |
| Średnia uncertainty: trafienie / błąd | 0,00260 / 0,00280 |
| Średnia uncertainty | 0,00266 |

W protokole wymagaliśmy korelacji co najmniej 0,25. Wynik 0,054 oznacza, że głowa uncertainty praktycznie nie odróżnia przykładów poprawnych od błędnych.

Widać też lukę train–test: Brier uncertainty na treningu spadł z 0,248 na epoce 20 do 0,00068 na epoce 200, natomiast na teście wynosi 0,300. To nie jest dobra kalibracja, lecz silne niedopasowanie sygnału treningowego do generalizacji.

## Przyczyny potwierdzone przez kod i logi

### 1. Self-referencyjny target `hard_error` zapada się do zera

Aktualna strata definiuje etykietę uncertainty jako `argmax(logits) != target` na tym samym minibatchu, na którym model jest uczony. Pod koniec runu accuracy treningowa to około 99,9%, dlatego prawie wszystkie etykiety są zerowe. MSE uczy głowę uncertainty przewidywania niemal zera.

To dokładnie zgadza się ze średnim uncertainty 0,00266 i niemal zerowym Brier na treningu. Na nowych obrazach błędy wciąż występują, ale głowa nie dostała generalizującego sygnału ryzyka.

### 2. Klasyfikacja nie przekazuje gradientu do decyzji uncertainty o liczbie grup

`GlobalBudgetAllocator` przyznaje dodatkowe grupy pętlą z twardym `argmax` po `stage_uncertainty`. Straight-through estimator jest zastosowany tylko do wyboru grup **wewnątrz** etapu przez utility. Nie zmiękcza ani nie różniczkuje liczby grup przyznawanej etapowi przez uncertainty.

W rezultacie gradient klasyfikacyjny może poprawiać utility, lecz nie uczy bezpośrednio decyzji: „który etap ma dostać dodatkową grupę?”. Uncertainty jest uczone niemal wyłącznie przez opisany wyżej Brier.

### 3. Jedna etykieta błędu nie mówi, który etap powinien dostać budżet

Brier uśrednia uncertainty czterech etapów i porównuje ją z jedną binarną etykietą poprawny/błędny obraz. Natomiast alokator podejmuje decyzję na podstawie *różnic* między uncertainty etapów. Strata nie zapewnia etapowej superwizji typu: „dodatkowe obliczenie w tym etapie obniżyłoby stratę najbardziej”.

To jest niespójność między tym, czego uczymy, a tym, co później robi alokator. Nie da się jej usunąć samą zmianą współczynnika Brier.

## Prawdopodobne, lecz jeszcze niepotwierdzone czynniki

### Twarda koncentracja budżetu

Po zapewnieniu jednej grupy na etap pozostałe grupy są kolejno dokładane do aktualnie najwyżej ocenionego etapu, aż do jego limitu. Przy 40 grupach i 4 etapach reguła sprzyja zapełnianiu jednego etapu przed przejściem do kolejnego. Słabo skalibrowany score może więc zbyt mocno wyciszyć ważny etap. Trzeba to potwierdzić histogramem `keep_count` per etap i per obraz.

### Zbyt uboga informacja dla polityki

Policy przewiduje utility i uncertainty jedynie z globalnego pooling'u 64-kanałowego stemu, jeszcze przed powstaniem cech semantycznych późnych etapów. To jest cena jednoprzejściowości; hipoteza wymaga osobnego ablation.

## Czego nie wolno wnioskować

- Nie obaliliśmy ogólnej idei uncertainty-aware allocation. Odrzucona została ta konkretna implementacja, strata i alokator.
- Nie robimy trzech seedów tej wersji ani Tiny ImageNet, ponieważ pełny seed 42 nie przeszedł z góry ustalonego progu.
- Różnicy latency z audytu nie wolno traktować jako speedupu: bramy mnożą aktywacje, lecz nie omijają fizycznie konwolucji.
- CIFAR-100 test był monitorowany co epokę. Użyliśmy ustalonej epoki 200, ale następny protokół powinien wydzielić walidację z danych treningowych i testować końcowo tylko raz.

## Plan naprawy przed następnym pełnym runem

1. Wydzielić train/validation/test; test zachować dla pojedynczego audytu końcowego.
2. Zastąpić `hard_error` targetem, który nie zanika wraz z accuracy treningową: porównać soft error, niezgodność dwóch augmentacji oraz out-of-fold/EMA teacher.
3. Uczyć **etapowej wartości dodatkowego obliczenia**: na części minibatchów mierzyć counterfactual wzrost straty po ograniczeniu konkretnego etapu. To daje prawdziwy target dla decyzji budżetowej.
4. Zmiękczyć przydział między etapami podczas treningu (soft quota, Gumbel-Sinkhorn lub straight-through również dla liczby grup). W inferencji zachować dokładny twardy budżet.
5. Logować histogram `keep_count`, entropy rozkładu budżetu i korelację score etapu z counterfactual wzrostem straty.
6. Zrobić 20–30-epokowy pilot na jednym seedzie. Dopiero po braku regresji względem utility-only, korelacji uncertainty–błąd >= 0,25 i niekolapsującym rozkładzie score uruchomić kolejne pełne 200 epok.

## Artefakty

- Audyt sparowany: `results/cifar100/global_publication/seed42/audit.json`.
- Diagnostyka UCLA: `results/cifar100/global_publication/seed42/ucla/uncertainty_report.json`.
- Historie: `results/cifar100/global_publication/seed42/{ucla,utility_only}/history.json`.
- Konfiguracje: `configs/cifar100_global_{ucla,utility}_publication.yaml`.

## Konkluzja

Eksperyment falsyfikuje obecną tezę w sposób kontrolowany: przy tym samym koszcie utility-only wygrywa, a uncertainty nie jest skalibrowane i nie jest dostatecznie związane z decyzją o przydziale budżetu. Następny wkład badawczy powinien uczyć **etapowej wartości dodatkowego obliczenia**, a nie dodawać ogólną predykcję prawdopodobieństwa błędu do obecnego twardego alokatora.
