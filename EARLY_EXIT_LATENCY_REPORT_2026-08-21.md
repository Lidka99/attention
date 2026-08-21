# Early-exit latency — batch size 1, CUDA

## Metoda

Benchmark uruchomiono na checkpointcie Tiny ImageNet early-exit seed 42.
Prefix-forward kończy wykonanie dokładnie po wybranym exit, bez uruchamiania
późniejszych bloków. Pomiar: 20 warm-up, 60 powtórzeń, CUDA synchronize po
każdym forwardzie.

| Exit | Mediana | P95 |
|---|---:|---:|
| stage 2 | 2,63 ms | 2,94 ms |
| stage 3 | 4,62 ms | 5,06 ms |
| stage 4 / full depth | 5,65 ms | 5,79 ms |

## Interpretacja

Held-out policy seed 42 kończyła 15,06% obrazów po stage 2, 36,18% po stage
3, a 48,76% wykonywała do końca. Ważona oczekiwana mediana wynosi około 4,82
ms, czyli około 14,6% mniej niż pełne 5,65 ms. Jest to zgodne z proxy kosztu
84,75% i potwierdza, że adaptive-depth może dawać rzeczywisty zysk czasu.

To nie jest końcowy benchmark publikacyjny: trzeba powtórzyć go dla wszystkich
seedów, batch size 1 i większego batcha, po rozgrzaniu oraz na określonym
sprzęcie. Dynamiczne batchowanie może zmienić wynik, dlatego raportujemy
prefix latency oddzielnie od end-to-end policy latency.
