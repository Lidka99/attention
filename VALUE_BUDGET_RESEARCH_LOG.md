# Dziennik nowego kierunku: stage value-of-compute

Stan na 2026-08-06. Ten dokument opisuje następcę nieudanego global UCLA.

## Co badamy

Nie uczymy już ogólnej uncertainty obrazu. Nowa hipoteza brzmi: przy stałym
globalnym budżecie kanałów model powinien przewidywać, **w którym etapie jedna
dodatkowa porcja obliczeń jest najbardziej wartościowa**. Target jest
kontrfaktyczny: mierzymy wzrost straty po ograniczeniu danego etapu.

## Gdzie jest kod

| Element | Plik | Rola |
|---|---|---|
| Model | `src/attentionv3/models/resnet_value_budget.py` | `GlobalValueBudgetResNet50`; 4 etapy, 16 grup/etap, dokładny budżet. |
| Target i strata wartości | `src/attentionv3/training/value_targets.py` | Zamienia kontrfaktyczne straty na rozkład targetu oraz liczy `stage_value_loss`. |
| Loader z walidacją | `src/attentionv3/data/cifar100.py` | Deterministyczny split 45k train / 5k validation / 10k test. |
| Pilot | `experiments/06_cifar100_value_budget_pilot.py` | DDP, kontrfaktyczne ablacjami, logi wartości i alokacji. |
| Konfiguracja value | `configs/cifar100_value_budget_pilot.yaml` | Value-of-compute, minimum 4 grupy/etap, bez testu w fazie diagnostycznej. |
| Konfiguracja kontrolna | `configs/cifar100_utility_floor_pilot.yaml` | Utility-only z identycznym splitem i floor. |
| Testy | `tests/test_value_targets.py`, `tests/test_resnet_ucla.py`, `tests/test_cifar100.py` | Target, exact budget/floor oraz split danych. |

Legacy UCLA i jego wynik negatywny pozostają nienaruszone. Opis: 
`FAILURE_REPORT_GLOBAL_UCLA_SEED42.md`; pełniejsze wyniki:
`EXPERIMENT_RESULTS.md`.

## Co już ustaliliśmy

### 1. Global UCLA — odrzucona wersja

- wynik końcowy: UCLA 69,82% vs utility-only 71,58% na CIFAR-100;
- różnica UCLA − utility-only: −1,76 pp, 95% CI [−2,58; −0,93];
- uncertainty–błąd: korelacja 0,054.

Wynik i przyczyny: `FAILURE_REPORT_GLOBAL_UCLA_SEED42.md`.

### 2. Pierwszy value pilot — tylko techniczny

`results/cifar100/value_budget_pilot/seed42/` oraz przypadkowo
`results/cifar100/`.

Pierwotny zachłanny alokator zapadał się do około `[16, 13, 10, 1]` grup.
Wyników nie używamy do porównań naukowych.

### 3. Value pilot z floor

`results/cifar100/value_budget_pilot/seed42_stage_floor/`

- test: 58,15%;
- wszystkie etapy dostały co najmniej 4 grupy;
- kontrola utility-only z tym samym floor: 58,80% test,
  `results/cifar100/utility_floor_pilot/seed42/`.

To poprawiło kolaps, ale value nie pokonało utility.

### 4. Diagnostyczny value pilot

`results/cifar100/value_budget_diagnostic/seed42/`

- 20/20 epok, validation 60,68%; test celowo nie był liczony;
- target entropy 0,288: kontrfaktyczny target jest informacyjny;
- top-1 agreement value vs target 14,8%: głowa wartości jeszcze go nie
  odtwarza;
- mean keep `[4, 16, 16, 4]`: floor działa, ale zachłanny alokator wciąż
  zamienia score w skrajną decyzję.

## Dlaczego obecny mechanizm nie wystarcza

Wartości etapów mogą różnić się subtelnie, ale greedy `argmax` wypełnia etap
o najwyższym score do limitu 16, potem kolejny. Przez to nawet niewielka
różnica score staje się decyzją binarną: minimum albo maksimum. To niszczy
informację z wartości i utrudnia uczenie policy.

## Następny plan implementacyjny

1. Zastąpić greedy allocation przez **soft quota + exact rounding**:
   `4 + 24 * softmax(stage_values / temperature)`, następnie capped
   largest-remainder rounding do dokładnie 40 grup.
2. Zalogować temperaturę, entropy alokacji i faktyczne quota per etap.
3. Zwiększyć `value_weight` i częstotliwość kontrfaktycznych targetów tylko
   po sprawdzeniu, że soft quota przestaje dawać skrajne `[4,16,16,4]`.
4. Uruchomić krótki pilot **wyłącznie z validation**, bez testu.
5. Jeśli agreement i stabilność alokacji rosną, uruchomić utility-only z
   identycznym soft quota.
6. Dopiero po wygraniu na validation: jeden końcowy test CIFAR-100, trzy
   seedy, Tiny ImageNet, a później ConvNeXt V2.

## Ważne zasady

- Test CIFAR-100 nie służy już do strojenia kolejnych pilotów.
- Nie deklarować realnego speedupu: obecne bramy mnożą aktywacje, lecz nie
  omijają fizycznie konwolucji.
- Muon jest przyszłą ablacją optymalizatora, nie rozwiązaniem sygnału value.
- Przed każdym runem robić commit; przy nowym treningu podawać użytkownikowi
  jedną pełną komendę uruchomienia oraz komendę `watch`.
