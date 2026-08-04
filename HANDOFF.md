# Handoff — attentionv3

Ten plik pozwala bezpiecznie wznowić pracę po utracie kontekstu rozmowy.

## Cel projektu

Badanie kompresji CNN przez channel attention. Docelowa hipoteza: przy stałym
globalnym budżecie grup kanałów, uncertainty-aware attention może lepiej
alokować obliczenia między warstwami ResNet-50 niż dynamiczna utility-only
attention i static pruning.

Ważne: obecne maskowanie wycisza aktywacje, ale jeszcze nie omija fizycznie
konwolucji. Nie deklarować realnego speedupu przed strukturalnym wykonaniem.

## Dokumenty

- `README.md` — uruchamianie projektu;
- `EXPERIMENT_RESULTS.md` — zakończone wyniki wcześniejszego, lokalnego UCLA;
- `PUBLICATION_PROTOCOL.md` — zamrożone zasady eksperymentu publikacyjnego;
- ten plik — operacyjny stan i kolejne kroki.

## Stan kodu

Najważniejsze commity:

```text
dc9d7d7 add reproducible cifar attention experiments
678c284 add tiny imagenet data adapter
6f0c674 add publication audit for cifar checkpoints
db64902 integrate global budget attention in resnet
2e535aa add global attention smoke configurations
cf30f96 add frozen global attention publication protocol
```

Kluczowe komponenty:

- `GlobalBudgetAllocator`: dokładnie stały wspólny budżet dla 4 etapów;
- `GlobalUCLAResNet50`: jednoprzejściowy model z polityką po stemie ResNet;
- `global UCLA`: budżet między warstwami alokowany przez uncertainty;
- `global utility-only`: kontrola alokująca budżet przez utility;
- straight-through gate zapewnia gradient do utility;
- `experiments/05_cifar100_publication_audit.py`: bootstrap CI, porównania
  sparowane, per-class accuracy, parametry i latency.

Testy przed kontynuacją:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Ostatni znany wynik: 31/31 testów OK.

## Dane lokalne

- CIFAR-100: `/home/users/s224574/attention/attention_v2/data`;
- Tiny ImageNet: `data/tiny-imagenet-200` (ignorowane przez Git);
- dane i wyniki są ignorowane przez `.gitignore`.

## Aktualny run

Pełny, 200-epokowy global UCLA CIFAR-100, seed 42, na 4 GPU:

```text
config: configs/cifar100_global_ucla_publication.yaml
output: results/cifar100/global_publication/seed42/ucla/
```

Monitorowanie z katalogu repozytorium:

```bash
watch -n 5 'tail -n 35 results/cifar100/global_publication/seed42/ucla/history.json'
watch -n 2 nvidia-smi
```

Nie uruchamiać kolejnego treningu na GPU przed sprawdzeniem, czy ten proces się
zakończył (`nvidia-smi` i ostatnia epoka w `history.json`).

## Zamrożony protokół publikacyjny

- dataset: CIFAR-100;
- 4 GPU DDP, batch size 128/GPU;
- 200 epok: 20 warm-up, 120 sparsification, 60 fine-tuning;
- learning rate: 1e-3 / 5e-4 / 1e-4;
- 16 grup na etap, 4 etapy, globalny budżet 62,5% = dokładnie 40 z 64 grup;
- seedy: 42, 123, 2026;
- kontrola: global utility-only z identycznym protokołem.

## Kolejność dalszych prac

1. Dokończyć global UCLA seed 42 i zapisać wynik.
2. Uruchomić global utility-only seed 42:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  experiments/03_cifar100_pilot.py \
  --config configs/cifar100_global_utility_publication.yaml \
  --output-dir results/cifar100/global_publication/seed42/utility_only
```

3. Uruchomić `experiments/05_cifar100_publication_audit.py` na obu checkpointach
   oraz sprawdzić Top-1, CI, ECE/NLL/Brier i uncertainty–błąd.
4. Tylko jeśli UCLA spełnia kryterium go/no-go z `PUBLICATION_PROTOCOL.md`,
   powtórzyć oba warianty dla seedów 123 i 2026.
5. Dodać globalne konfiguracje/pilot Tiny ImageNet; adapter jest gotowy w
   `src/attentionv3/data/tinyimagenet.py`.
6. Dodać CIFAR-100-C jako ewaluację corruption bez ponownego treningu.
7. Dopiero po walidacji metody wdrożyć strukturalne wykonanie/pruning i
   raportować realną latencję.

### Komendy po zakończeniu utility-only seed 42

Podstawowy audyt porównuje końcowe checkpointy po 200 epokach (nie `best.pt`):

```bash
PYTHONPATH=src python experiments/05_cifar100_publication_audit.py \
  --run global_ucla=configs/cifar100_global_ucla_publication.yaml,results/cifar100/global_publication/seed42/ucla/latest.pt \
  --run utility_only=configs/cifar100_global_utility_publication.yaml,results/cifar100/global_publication/seed42/utility_only/latest.pt \
  --reference utility_only \
  --output results/cifar100/global_publication/seed42/audit.json
```

Następnie diagnostyka uncertainty dla UCLA:

```bash
PYTHONPATH=src python experiments/04_cifar100_uncertainty_report.py \
  --config configs/cifar100_global_ucla_publication.yaml \
  --checkpoint results/cifar100/global_publication/seed42/ucla/latest.pt \
  --output-dir results/cifar100/global_publication/seed42/ucla
```

## Ustalony kierunek po obecnym treningu (2026-08-02)

Nie zmieniamy obecnego, zamrożonego protokołu w trakcie jego wykonania.
Po zakończeniu pary global UCLA vs global utility-only na CIFAR-100:

1. Pierwszym nowoczesnym backbone'em będzie **ConvNeXt V2-Tiny**. Jest to
   kontrolowane rozszerzenie CNN: globalny budżet nadal może alokować grupy
   kanałów/bloków między etapami. Cel: sprawdzić, czy mechanizm generalizuje
   poza ResNet-50, przy tym samym rygorze kontroli i kosztu.
2. **MambaVision lub VMamba** traktujemy jako drugi, eksploracyjny kierunek.
   Hipoteza: globalny, uncertainty-aware budżet może przydzielać zasoby między
   bloki state-space. Jest to bardziej nowe, ale wymaga osobnego projektu
   mechanizmu i uczciwego pomiaru rzeczywistego kosztu, więc nie mieszamy go z
   podstawową walidacją CNN.

Potencjalny wkład pracy to nie samo użycie nowszego backbone'u, lecz
`uncertainty-calibrated global compute budgeting for efficient visual
backbones`: dokładny globalny budżet, alokacja zależna od obrazu, kontrola
utility-only/static oraz późniejsze strukturalne wykonanie i pomiar latency.

## Wyniki, których nie należy nadinterpretować

Wcześniejszy lokalny UCLA na CIFAR-100 w trzech seedach nie pokonał static i
był niestabilny. Wyniki są w `EXPERIMENT_RESULTS.md`. Globalny mechanizm ma
tylko 10-epokowe smoke runy — nie są wynikami publikacyjnymi.
