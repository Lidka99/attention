# attentionv3

Nowy projekt eksperymentów nad dynamiczną, niepewnością sterowaną atencją i adaptacyjnym budżetem obliczeniowym na ImageNet.

Główny plan badawczy znajduje się w [PLAN_BADAWCZY.md](PLAN_BADAWCZY.md). Implementacja powinna rozpocząć się od pilota na ImageNet-100 i adaptera ResNet-50, a dopiero później przejść do pełnego ImageNet-1K.

Status: koncepcja i protokół badawczy.


## Trening na czterech GPU

Pilot ImageNet-100 jest uruchamiany przez DDP; konfiguracja wymaga czterech procesów/GPU:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  experiments/01_imagenet100_pilot.py --config configs/imagenet100_resnet50.yaml
```

Każdy proces dostaje własny fragment danych przez `DistributedSampler`; tylko rank 0 zapisuje manifest, checkpoint i historię.
