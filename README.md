# attentionv3 — UCLA

Prototyp badawczy **Uncertainty-Calibrated Layerwise Attention (UCLA)** dla
ImageNet. Model ResNet-50 wybiera grupy kanałów po każdej z czterech warstw;
kontroler używa utility, a w wariancie UCLA również predykowanej niepewności,
aby przydzielić ograniczony dodatkowy budżet trudniejszym obrazom.

Projekt służy najpierw do pilota na ImageNet-100. Pełne założenia i kryteria
go/no-go są w [PLAN_BADAWCZY.md](PLAN_BADAWCZY.md), a zasady uczciwego
porównania trzech wariantów w [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Co jest zaimplementowane

- ResNet-50 z bramkami grup kanałów po `layer1`–`layer4`;
- trzy warianty: `ucla`, `utility_only` i `static`;
- straight-through estimator dla hard top-k;
- trzyfazowy harmonogram: warm-up, sparsification, fine-tuning;
- trening student–nauczyciel (nauczyciel jest opcjonalny);
- DDP dla dokładnie czterech GPU;
- kalibracja temperatury oraz Top-1/Top-5, NLL, Brier, ECE i latency.

## Wymagania

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

Do właściwego pilota potrzebne są 4 GPU CUDA oraz ImageNet ułożony jako
`ImageFolder`:

```text
<DATA_DIR>/train/<klasa>/*.JPEG
<DATA_DIR>/val/<klasa>/*.JPEG
```

Oficjalny ImageNet validation trzeba wcześniej rozłożyć do folderów klas.
Następnie zmień `data_dir` w używanym pliku `configs/imagenet100_*.yaml` na
`<DATA_DIR>`. Manifest stu klas jest tworzony przy pierwszym runie i zapisywany
w katalogu wyniku; nie mieszaj katalogów wyników między seedami.

## Najpierw: pilot seed 42

Na GPU 11 GB zacznij od `batch_size: 16` lub `32` **we wszystkich trzech
YAML-ach**. Domyślne `128` jest liczone na proces/GPU i może prowadzić do OOM.
Po sprawdzeniu pamięci można zwiększać tę wartość, zachowując ją identyczną dla
porównywanych metod.

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  experiments/01_imagenet100_pilot.py \
  --config configs/imagenet100_ucla.yaml \
  --output-dir results/imagenet100/seed42/ucla
```

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  experiments/01_imagenet100_pilot.py \
  --config configs/imagenet100_utility_only.yaml \
  --output-dir results/imagenet100/seed42/utility_only
```

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  experiments/01_imagenet100_pilot.py \
  --config configs/imagenet100_static.yaml \
  --output-dir results/imagenet100/seed42/static
```

Każdy run tworzy `history.json`, checkpoint końcowy `latest.pt`, checkpoint
najlepszej walidacji `best.pt` i manifest klas. Po
każdej epoce sprawdź `val_top1`, `keep` oraz fazę w `history.json`. Oczekiwane
fazy to pięć epok `warmup`, 20 `sparsification`, a następnie 10 `fine_tune`.

Opcjonalny `--teacher-checkpoint <plik.pt>` należy podać identycznie dla
wszystkich wariantów. Bez niego składnik distillation ma wartość zero, więc
porównanie nadal jest poprawne, jeśli żaden wariant nie używa nauczyciela.

## Szybki eksperyment: CIFAR-100 na 4 GPU

Przed ImageNet warto zweryfikować cały pipeline na CIFAR-100. W tym środowisku
dane są już dostępne w konfiguracjach CIFAR; jeżeli zmienisz lokalizację, ustaw
`data_dir` w odpowiednich plikach `configs/cifar100_*.yaml`. Pilot ma 10 epok
(2 warm-up, 6 sparsification, 2 fine-tuning) i zapisuje wyniki w `results/`.

Każda z poniższych komend uruchamia **jeden eksperyment na wszystkich czterech
GPU jednocześnie**. Nie uruchamiaj trzech komend naraz: każdy run używa całego
zestawu GPU przez DDP.

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  experiments/03_cifar100_pilot.py \
  --config configs/cifar100_ucla.yaml \
  --output-dir results/cifar100/seed42/ucla
```

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  experiments/03_cifar100_pilot.py \
  --config configs/cifar100_utility_only.yaml \
  --output-dir results/cifar100/seed42/utility_only
```

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  experiments/03_cifar100_pilot.py \
  --config configs/cifar100_static.yaml \
  --output-dir results/cifar100/seed42/static
```

Po runie otwórz `history.json` w katalogu wyniku. Końcowe metryki znajdują się
w ostatnim wpisie `validation`: `accuracy` to Top-1, a `mean_keep_ratio` to
średni odsetek zachowanych grup kanałów. `latest.pt` zawiera checkpoint po
ustalonej liczbie epok, a `best.pt` checkpoint najwyższej walidacji wraz z
numerem epoki. W zamrożonym protokole publikacyjnym porównujemy `latest.pt`;
`best.pt` służy wyłącznie do diagnostyki, aby nie wybierać modelu na podstawie
zbioru testowego.
CIFAR-100 służy do walidacji implementacji; wyników nie należy porównywać
bezpośrednio z ImageNet ani traktować jako końcowego wyniku badawczego.

## Kalibracja i raport

Po ukończeniu runu uruchom osobno dla każdego checkpointu:

```bash
PYTHONPATH=src python experiments/02_calibrate_and_report.py \
  --config configs/imagenet100_ucla.yaml \
  --checkpoint results/imagenet100/seed42/ucla/latest.pt \
  --output-dir results/imagenet100/seed42/ucla/evaluation
```

Raport `evaluation.json` zawiera metryki przed i po temperature scaling oraz
medianę/P95 latency. Kalibracja może zmienić NLL, Brier i ECE, ale nie Top-1
ani latency.

## Kolejność eksperymentów

1. Seed 42: uruchom trzy warianty i sprawdź stabilność logów.
2. Zapisz oraz porównaj raporty kalibracji.
3. Powtórz dokładnie te same runy dla seedów 123 i 2026.
4. Dopiero wtedy wyciągaj średnie i odchylenia między seedami.
5. Przed twierdzeniami o przewadze metody dodaj pełny ResNet-50 bez attention
   jako baseline.

## Ważne ograniczenia

Obecna maska zeruje aktywacje, ale nie omija obliczeń warstw konwolucyjnych.
`keep_ratio` nie jest więc speedupem, a realny latency może się nie poprawić.
Teza o przyspieszeniu wymaga później strukturalnego pruning albo kernelów
wykonujących tylko wybrane grupy kanałów.
