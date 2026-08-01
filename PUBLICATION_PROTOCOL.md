# Protokół publikacyjny — attentionv3

## Teza do sprawdzenia

Przy stałym, globalnym budżecie aktywnych grup kanałów, polityka attention
sterowana utility i uncertainty może lepiej alokować obliczenia między warstwy
splotowe niż static pruning i dynamiczna utility bez uncertainty.

To jest hipoteza falsyfikowalna, nie założony wynik.

## Status obecnej metody

Dotychczasowy lokalny UCLA nie potwierdził przewagi nad static pruning na
CIFAR-100. Wyniki są zapisane w `EXPERIMENT_RESULTS.md`. Nie wolno wybierać
hiperparametrów na podstawie najlepszego seeda ani przedstawiać obecnej wersji
jako metody wygrywającej.

## Wariant proponowany do oceny

`Global uncertainty-aware convolutional attention`:

- jeden, dokładnie stały budżet grup dla czterech etapów ResNet;
- alokator rozdziela ten budżet między warstwy dla każdego obrazu;
- utility wybiera grupy wewnątrz przydziału warstwy;
- uncertainty może zmienić *rozkład*, ale nie zwiększa całkowitego kosztu;
- decyzja jest raportowana per warstwa i per obraz.

Przed uruchomieniem zamrożonego protokołu wykonujemy wyłącznie 10-epokowy
smoke run na CIFAR-100, aby sprawdzić DDP, gradienty i dokładny globalny
budżet. Wynik smoke runu nie jest używany w tabeli publikacyjnej.

## Zamrożone kontrole

1. full ResNet-50 bez bramek;
2. static channel attention;
3. dynamic utility-only z tym samym globalnym budżetem;
4. global utility-only — kontrola samej globalnej alokacji;
5. global uncertainty-aware — metoda badana.

Wszystkie warianty używają tego samego backbone, transformacji, liczby epok,
optymalizatora, batch size i seedów.

## Metryki wymagane w każdym runie

- Top-1 i Top-5;
- średni, medianowy i P95 globalnego keep-ratio;
- budget per stage oraz jego rozkład dla obrazów;
- ECE, NLL, wieloklasowy Brier;
- Brier i korelacja uncertainty–błąd;
- stabilność maski dla dwóch augmentacji tego samego obrazu;
- median/P95 latency dopiero po wdrożeniu strukturalnego wykonania.

## Reguły eksperymentalne

- CIFAR-100: trzy seedy (42, 123, 2026), raport średnia ± SD;
- Tiny ImageNet: trzy seedy po zamrożeniu protokołu CIFAR;
- ImageNet-100: trzy seedy dopiero po spełnieniu kryterium go/no-go;
- bez zmian hiperparametrów po rozpoczęciu trzech seedów danego wariantu;
- kod, YAML, commit, wersje bibliotek i GPU zapisane przy każdym runie.

## Kryterium go/no-go

Przejście do Tiny ImageNet wymaga równocześnie:

1. global uncertainty-aware nie jest gorszy od global utility-only o więcej
   niż 0,5 pp Top-1 przy tym samym globalnym budżecie;
2. dodatnia i powtarzalna korelacja uncertainty–błąd (co najmniej 0,25);
3. lepsza stabilność masek albo lepsza jakość na najwyższym decylu uncertainty;
4. brak istotnie większej zmienności między seedami niż kontrole.

Niespełnienie kryterium jest wynikiem negatywnym i wskazuje na konieczność
zmiany mechanizmu, a nie na selekcję korzystnych runów.
