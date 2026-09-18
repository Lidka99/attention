# Plan Badawczy — Feedback-Driven Filter Pruning for Deep Convolutional Networks

## Metadane

- **Data:** 25.05.2026
- **Promotor:** (do uzupełnienia)
- **Stan początkowy:** Kod `self_attentive_conv.py` (TensorFlow) — statyczna atencja skalarna na filtr + L1, testowane na MNIST/Fashion-MNIST
- **Docelowy framework:** PyTorch (port z TF)
- **Docelowe zbiory:** CIFAR-10 → CIFAR-100 → ImageNet-100 → **ImageNet-1K**
- **Sprzęt:** Klaster GPU (umożliwia trening ResNet-50 na pełnym ImageNet)

---

## Pytania badawcze

| # | Pytanie |
|---|---|
| **RQ1** | Czy atencja warunkowana cechami wejściowymi (input-dependent) daje lepsze przycinanie niż statyczne wagi? |
| **RQ2** | Czy sprzężenie zwrotne z głębszych warstw poprawia dobór filtrów w płytszych warstwach? |
| **RQ3** | Czy można zautomatyzować dobór liczby filtrów bez ręcznego strojenia hiperparametrów? |
| **RQ4** | Jak różne mechanizmy atencji wpływają na kompromis accuracy-vs-FLOPs na ImageNet? |
| **RQ5** | Czy można połączyć przycinanie, aproksymację niskorzędową, kwantyzację i kompresję bezstratną w jeden pipeline bez utraty jakości? |

---

## Warianty mechanizmów atencji

### Wariant A: Input-Dependent Channel Attention (SE-like)
Rozszerzenie oryginalnej metody — atencja jest funkcją cech, nie statycznym parametrem.

```
conv(x) → GAP → FC → ReLU → FC → Sigmoid → wagi atencji → out = conv(x) * attn
```

- SEAttentiveConv2d w `src/layers/attentive_conv.py`
- Nowość: atencja zależy od zawartości mapy cech (per-sample), nie jest statyczna

### Wariant B: Feedback Cross-Layer Attention (KLUCZOWA INNOWACJA)
Sprzężenie zwrotne — cechy z warstwy L+1 modulują wagi atencji w warstwie L.

```
Warstwa L ──(features)──> Warstwa L+1 ──(features)──> Warstwa L+2
    ^                           |
    └───(feedback_attention)────┘
```

- FeedbackAttentiveConv2d w `src/layers/feedback_attention.py`
- Motywacja biologiczna: top-down attention w korze wzrokowej
- **Nikt nie użył tego do przycinania filtrów** — główna kontrybucja naukowa

### Wariant C: Gumbel-Softmax Discrete Pruning
Zamiast miękkiego przycinania L1 — **twarde decyzje** (filtr włączony/wyłączony) podczas treningu.

- GumbelAttentiveConv2d w `src/layers/gumbel_attention.py`
- Temperatura `tau` maleje w trakcie treningu (annealing od 5.0 do 0.5)
- Po treningu: `probs > 0.5` → filtr zostaje, w.p.p. usunięty

---

## Architektura eksperymentów

| Etap | Dataset | Architektura | Cel |
|---|---|---|---|
| **1. Proof of concept** | CIFAR-10 | ResNet-18 | Porównanie wariantów A/B/C z baseline'ami |
| **2. Ablation study** | CIFAR-100 | ResNet-34 | Który komponent daje najwięcej? |
| **3. Scaling** | ImageNet-100 (subset) | ResNet-34 | Walidacja na większym zbiorze |
| **4. Full scale** | **ImageNet-1K** | ResNet-50 | Główny wynik do publikacji |
| **5. Transfer** | COCO / VOC (opcjonalnie) | Pruned ResNet-50 | Czy przycięty model się transferuje? |

---

## Baselines do porównania

| Metoda | Opis |
|---|---|
| **Network Slimming** (Liu 2017) | Skalowanie kanałów przez γ w BatchNorm + L1 |
| **SE-Net** (Hu 2018) | Channel attention bez przycinania |
| **ECA-Net** (Wang 2020) | Lżejsza channel attention |
| **L1-norm pruning** | Przycinanie filtrów z najmniejszą normą L1 |
| **Taylor pruning** (Molchanov 2017) | Przycinanie na podstawie wpływu na loss |
| **Oryginalny AttentiveConv (A0)** | Statyczna atencja skalarna — port z TensorFlow |

---

## Metryki ewaluacji

- **Accuracy** (Top-1, Top-5 dla ImageNet)
- **# parametrów** przed i po przycięciu
- **FLOPs** (teoretyczne i rzeczywiste)
- **Inference time** (rzeczywisty pomiar na GPU)
- **Sparsity ratio** na warstwę
- **Rozmiar modelu po kompresji** (kwantyzacja INT8 + Huffman)

---

## Pipeline kompresji (etap końcowy)

```
Wytrenowany model z atencją
    ↓
Hard pruning (filtry z wagą < τ usunięte)
    ↓
Fine-tuning przyciętego modelu
    ↓
Low-rank SVD dekompozycja pozostałych filtrów
    ↓
Quantization-Aware Training (INT8)
    ↓
Weight clustering + Huffman coding
    ↓
Model gotowy do deploymentu
```

---

## Harmonogram (12–18 miesięcy)

| Miesiąc | Zadanie |
|---|---|
| **1–2** | Port do PyTorch, implementacja wariantu A na CIFAR-10 |
| **3–4** | Implementacja wariantu B (feedback attention) + eksperymenty CIFAR |
| **5–6** | Implementacja wariantu C (Gumbel-Softmax), ablation study na CIFAR-100 |
| **7–8** | Przygotowanie pipeline'u dla ImageNet, wstępne eksperymenty ImageNet-100 |
| **9–11** | Pełne eksperymenty ImageNet-1K na klastrze GPU |
| **12–13** | Pipeline kompresji (kwantyzacja + kodowanie) + analiza wyników |
| **14–16** | Pisanie publikacji + dodatkowe eksperymenty |

---

## Oczekiwane rezultaty (3 publikacje)

1. **"Feedback-Driven Attention for Automated Filter Pruning"** — konferencja (CVPR/ECCV/ICCV lub NeurIPS/ICML workshop)
2. **"End-to-End Compression Pipeline for Attention-Pruned CNNs"** — journal (IEEE TPAMI / Neurocomputing)
3. **Praca doktorska** — scalenie wszystkich wyników

---

## Kontrybucje naukowe

| # | Kontrybucja | Dlaczego nowe? |
|---|---|---|
| 1 | **Feedback Cross-Layer Attention** | Nikt nie użył sprzężenia zwrotnego z głębszych warstw do sterowania przycinaniem filtrów |
| 2 | **Trójfazowe porównanie mechanizmów atencji** | Systematyczne porównanie static vs SE vs feedback vs Gumbel — brak w literaturze |
| 3 | **Gumbel-Softmax discrete pruning** | Dyskretne decyzje w trakcie treningu zamiast post-hoc threshold |
| 4 | **Joint pruning + low-rank + quantization pipeline** | Łączenie przycinania, SVD i kwantyzacji w jednym pipeline |
| 5 | **Walidacja ImageNet-scale** | Większość prac o atencji dla przycinania testuje tylko CIFAR |

---

## Dotychczasowe wyniki (punkt startowy)

### MNIST (oryginalny kod TF)

| Beta | Accuracy | Filtry usunięte (z 144) |
|---|---|---|
| 0.0 | 99.08% | 0 (0%) |
| 0.1 | 98.87% | 64 (44%) |
| 0.2 | 98.74% | 75 (52%) |
| 0.5 | 98.69% | 91 (63%) |
| 1.0 | 98.40% | 103 (72%) |

### Fashion-MNIST (oryginalny kod TF)

| Beta | Accuracy | Filtry usunięte (z 144) |
|---|---|---|
| 0.0 | 88.35% | 0 (0%) |
| 0.1 | 89.36% | 53 (37%) |
| 0.2 | 88.49% | 78 (54%) |
| 0.5 | 86.87% | 98 (68%) |
| 1.0 | 85.24% | 111 (77%) |

---

## Struktura projektu `attention_v2/`

```
attention_v2/
├── configs/
│   └── cifar10_resnet18.yaml       # Konfiguracja YAML
├── src/
│   ├── layers/
│   │   ├── attentive_conv.py       # OriginalAttentiveConv2d + SEAttentiveConv2d + fabryka
│   │   ├── feedback_attention.py   # FeedbackAttention + FeedbackAttentiveConv2d
│   │   └── gumbel_attention.py     # GumbelAttentiveConv2d z annealingiem tau
│   ├── models/
│   │   ├── resnet_cifar.py         # ResNet-18/34 dla CIFAR (32x32)
│   │   └── resnet_imagenet.py      # ResNet-18/34/50 dla ImageNet (224x224)
│   ├── training/
│   │   ├── trainer.py              # ThreePhaseTrainer (warm-up → fine-tune → final)
│   │   └── losses.py               # L1, Gumbel sparsity, FLOPs-constrained, entropy
│   ├── pruning/
│   │   ├── hard_prune.py           # Fizyczne usuwanie filtrów + maski
│   │   └── low_rank.py             # SVD dekompozycja warstw
│   ├── compression/
│   │   ├── quantize.py             # QAT + INT8
│   │   └── huffman.py              # Kodowanie Huffmana
│   └── utils.py                    # Wizualizacja, FLOPs, metryki
├── experiments/
│   ├── 01_cifar10_baseline.py      # Port oryginału z TF → PyTorch na CIFAR-10
│   ├── 02_cifar10_variant_a.py     # SE-Attention ResNet-18
│   ├── 03_cifar10_variant_b.py     # Feedback Cross-Layer Attention
│   ├── 04_cifar10_variant_c.py     # Gumbel-Softmax Discrete Pruning
│   └── 05_cifar10_all.py           # Porównanie wszystkich + standard ResNet
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Jak odpalić

```bash
cd attention_v2
pip install -r requirements.txt

# 1. Baseline — port oryginału (najszybszy, ~10 min na GPU):
python experiments/01_cifar10_baseline.py

# 2. Wariant A — SE-Attention (~30 min):
python experiments/02_cifar10_variant_a.py

# 3. Wariant B — Feedback Attention:
python experiments/03_cifar10_variant_b.py

# 4. Wariant C — Gumbel-Softmax:
python experiments/04_cifar10_variant_c.py

# 5. Pełne porównanie:
python experiments/05_cifar10_all.py
```

---

## Uwagi do dalszej pracy

1. **Eksperymenty** na CIFAR-10 najpierw — szybka iteracja, niski koszt
2. **Feedback attention (Wariant B)** to największa innowacja — warto mu poświęcić najwięcej uwagi
3. **Gumbel-Softmax** wymaga strojenia harmonogramu temperatury — eksperymentuj z `start_tau` i `end_tau`
4. **ImageNet** wymaga `resnet_imagenet.py` + `attentive_resnet50(attention_type='...')` — kod już jest gotowy
5. **Aproksymacja low-rank** (`src/pruning/low_rank.py`) — użyj po przycięciu filtrów, przed fine-tuningiem
6. **Kwantyzacja** (`src/compression/quantize.py`) — wymaga PyTorch >= 2.0 z backendem fbgemm
7. **Publikacja** — celuj w CVPR/ECCV (computer vision) lub NeurIPS (ML). Format: 8 stron + suplement
