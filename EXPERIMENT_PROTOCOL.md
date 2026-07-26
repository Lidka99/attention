# Protokół ablation study — ImageNet-100

## Cel

Porównać trzy metody przy identycznym zbiorze klas, seedzie, ResNet-50, liczbie epok, harmonogramie i czterech GPU:

| Konfiguracja | Co jest mierzone |
|---|---|
| `imagenet100_ucla.yaml` | utility + uncertainty + adaptacyjny dodatkowy budżet |
| `imagenet100_utility_only.yaml` | dynamiczna utility bez uncertainty i bez dodatkowego budżetu |
| `imagenet100_static.yaml` | jedna uczona, statyczna maska kanałów dla wszystkich obrazów |

## Zasady uczciwego porównania

- Nie zmieniać manifestu klas między wariantami: używać osobnego katalogu wyniku tylko przy tym samym `seed`.
- Dla finalnego wyniku uruchomić co najmniej trzy seedy: 42, 123, 2026.
- Używać tego samego checkpointu nauczyciela we wszystkich wariantach, albo konsekwentnie wyłączyć nauczyciela we wszystkich.
- Po treningu uruchomić kalibrację na osobnym splitcie walidacji i raportować surowe oraz skalibrowane wyniki.
- Nie interpretować keep-ratio jako speedup; do tezy o szybkości używać mediany i P95 latency.

## Kolejność

1. Wykonać jeden pilot seeda 42 dla wszystkich trzech konfiguracji.
2. Sprawdzić logi, czy faktyczny keep-ratio i phases są zgodne z YAML.
3. Dopiero potem uruchomić dwa kolejne seedy i finalną ewaluację.
