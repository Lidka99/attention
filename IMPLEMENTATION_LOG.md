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
