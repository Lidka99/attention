# Dziennik implementacji

## Commit 1 — prototyp kontrolera budżetu

### Co zostało zrobione

1. Dodano `BudgetController`, który łączy utility z uncertainty i wybiera grupy metodą hard top-k.
2. Dodano bazowy budżet oraz ograniczony dodatkowy budżet dla próbek o wysokiej niepewności.
3. Dodano `UCLAChannelAttention` jako adapter dla map cech CNN.
4. Dodano konfigurację pilota ImageNet-100.
5. Dodano cztery testy jednostkowe obejmujące budżet, adaptację, kształty oraz walidację grup.

### Ważne ograniczenie

Hard top-k jest na razie kontraktem inferencyjnym i nie zapewnia jeszcze gradientu do uczenia kontrolera. W następnym etapie trzeba dodać differentiable relaxation albo straight-through estimator oraz test, że loss może aktualizować głowy utility/uncertainty.

### Weryfikacja

```text
PYTHONPATH=src python -m unittest discover -s tests -v
Ran 4 tests ... OK
```

## Commit 2 — uczenie kontrolera przez straight-through estimator

### Co zostało zrobione

1. Dodano sigmoidową relaksację maski wokół progu top-k.
2. W trybie treningowym forward nadal zwraca dokładną maskę binarną, ale backward korzysta z relaksacji.
3. Dodano test sprawdzający gradienty do utility i uncertainty.

### Interpretacja

To rozwiązuje blokadę z pierwszego etapu, ale nie jest jeszcze pełnym treningiem ImageNet. Następny krok to podłączenie modułu do bloku ResNet oraz dodanie strat klasyfikacji, distillation, Brier i budżetu.

## Commit 3 — integracja z ResNet-50

### Co zostało zrobione

1. Dodano `UCLAResNet50` oparty na torchvision ResNet-50.
2. Wstawiono channel attention po `layer1`–`layer4`.
3. Model zwraca logits oraz diagnostykę budżetu z każdej warstwy.
4. Dodano testy forwardu ImageNet oraz przepływu gradientu do głowy utility.

### Ograniczenie

Maska zeruje aktywacje, ale nie daje jeszcze fizycznego speedupu — do tego potrzebne będzie grupowe kernel execution albo strukturalny hard pruning. Latency nie będzie deklarowane na podstawie samego proxy FLOPs.

## Commit 4 — funkcje strat treningowych

### Co zostało zrobione

1. Dodano cross-entropy jako główny cel klasyfikacyjny.
2. Dodano KL distillation z temperaturą i odłączonym nauczycielem.
3. Dodano Brier score, gdzie uncertainty przewiduje prawdopodobieństwo błędu.
4. Dodano karę za odchylenie średniej liczby aktywnych grup od budżetu.
5. Dodano `UCLALossBreakdown`, aby każdy składnik był logowany osobno.

### Ograniczenie

Target Brier jest obecnie etykietą błędu predykcji studenta i nie zastępuje cross-entropy. W pełnym treningu trzeba dodać osobny calibration split oraz kalibrację po treningu.

## Commit 5 — pętla treningu student–nauczyciel

### Co zostało zrobione

1. Dodano pętle `train_one_epoch` i `evaluate` dla standardowego `DataLoader`.
2. Nauczyciel jest zamrażany i używany tylko do distillation.
3. Każda epoka raportuje accuracy, wszystkie części straty oraz średni keep-ratio.
4. Dodano test integracyjny na syntetycznych danych.

### Granica tego etapu

Pętla nie pobiera danych ani nie wybiera klas ImageNet-100. Te decyzje należą do osobnego modułu danych, aby protokół podziału był jawny i reprodukowalny.

## Commit 6 — dane ImageNet-100 i skrypt pilota

### Co zostało zrobione

1. Dodano deterministyczny wybór 100 klas i manifest JSON z nazwami klas.
2. Dodano filtrowanie ImageFolder oraz jednoznaczne remapowanie etykiet.
3. Dodano skrypt pilota student–nauczyciel, który zapisuje historię i checkpoint po każdej epoce.
4. Dodano testy manifestu i mapowania klas bez potrzeby posiadania ImageNet.

### Warunek uruchomienia

Skrypt zakłada strukturę `data_dir/train/<klasa>` i `data_dir/val/<klasa>`. Oficjalny ImageNet validation wymaga wcześniejszego uporządkowania obrazów do folderów klas albo własnego adaptera etykiet.

## Commit 7 — kalibracja i raport końcowy

### Co zostało zrobione

1. Dodano stratyfikowany podział validation na calibration i held-out test.
2. Dodano temperature scaling dopasowywany wyłącznie na calibration split.
3. Dodano Top-1, Top-5, NLL, wieloklasowy Brier, ECE oraz latency median/P95.
4. Dodano skrypt raportu zapisujący surowe i skalibrowane metryki do JSON.

### Interpretacja

Kalibracja może poprawić NLL/ECE, lecz nie powinna zmieniać Top-1 ani latency modelu. Wszystkie trzy grupy metryk trzeba raportować osobno.
