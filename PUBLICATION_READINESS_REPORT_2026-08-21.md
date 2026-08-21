# attentionv3 — raport badań i gotowości publikacyjnej

**Data:** 2026-08-21  
**Status:** channel-budget routing zamknięty jako wynik negatywny; adaptive
depth / early exit jest w fazie replikacji. Seed 42 jest aktualnie trenowany.

## Streszczenie

Projekt badał dynamiczne oszczędzanie obliczeń w ResNet-50. Kolejno oceniono
uncertainty-aware channel routing, kontrfaktyczne value-of-compute, destylację
decyzji oracle'a oraz adaptive depth. Pierwsze trzy linie nie uzyskały
generalizującej przewagi nad prostymi kontrolami. Adaptive depth jako pierwszy
kierunek przechodzi pilot oracle i held-out confidence policy.

Nie należy deklarować realnego speedupu dla channel gates: obecny kod zeruje
aktywacje, lecz nie omija konwolucji. Early exit może pomijać całe późniejsze
bloki, dlatego ma wyższy potencjał systemowy.

## Standard metodologiczny

- porównywane modele otrzymują ten sam backbone, budget, dane, optimizer i seed;
- utility-only i static są podstawowymi kontrolami;
- globalny budget kanałów wynosi dokładnie 40 z 64 grup, z floor cztery grupy;
- piloty są oddzielane od zamrożonych wyników; nie wybierano najlepszego seeda;
- kalibracja progów early-exit jest rozdzielona od held-out oceny;
- aktualnie 42 testy jednostkowe przechodzą.

## UCLA uncertainty-aware routing

Zamrożony CIFAR-100, 200 epok, seed 42:

| Wariant | Top-1 | Keep-ratio |
|---|---:|---:|
| global UCLA | 69,82% | 62,5% |
| global utility-only | 71,58% | 62,5% |

Sparowana różnica UCLA minus utility-only to −1,76 pp, CI 95%
[−2,58; −0,93]. Korelacja uncertainty–błąd wyniosła 0,054. Teza, że ogólna
uncertainty jest dobrym sygnałem globalnej alokacji kanałów, została odrzucona.

## Kontrfaktyczne value-of-compute

Soft quota usunęło kolaps greedy allocation i zachowało exact budget, lecz
nie dało przewagi końcowej Tiny ImageNet po 100 epokach:

| Seed | Value | Utility-only | Różnica |
|---:|---:|---:|---:|
| 42 | 58,36% | 58,33% | +0,03 pp |
| 123 | 58,74% | 59,33% | −0,59 pp |
| średnia | 58,55% | 58,83% | −0,28 pp |

Wczesny dodatni pilot 20-epokowy nie utrzymał się po zbieżności. Głowa policy
potrafiła zapamiętać zamrożony batch, ale gęstszy target co batch dał agreement
tylko 7,19%. Problem nie był prostym brakiem gradientu, lecz niestacjonarnym
targetem i ograniczoną obserwowalnością wartości przyszłego obliczenia.

## Oracle channel transfers i routing

Label-informed oracle lokalnych transferów na 512 obrazach Tiny ImageNet
podniósł Top-1 o +10,94 pp z checkpointu value oraz +11,91 pp z utility-only.
Action space ma więc potencjał, lecz student nie potrafi go przewidzieć:

| Cechy i action space | Held-out accuracy | Majority | Regret |
|---|---:|---:|---:|
| stem, 13 akcji globalnych | 20,90% | 23,83% | 1,534 NLL |
| layer1, przyszłe etapy 2–4 | 31,64% | 33,98% | 1,410 NLL |
| layer2, przyszłe etapy 3–4 | 59,77% | 61,13% | 0,565 NLL |

Globalny i sekwencyjny channel routing jest zamknięty jako główny kierunek.
Pozostaje wartościowym wynikiem negatywnym, ale nie podstawą pozytywnej pracy.

## Adaptive depth / early exit

Pilot Tiny ImageNet, seed 123, 20 epok, z głowami po stage 2, stage 3 i stage 4:

| Metryka | Wynik |
|---|---:|
| stage 2 / stage 3 / stage 4 Top-1 | 35,67% / 52,53% / 54,22% |
| label-informed oracle Top-1 | 62,53% |
| oracle mean cost fraction | 76,05% |

Na osobnej połowie walidacji policy confidence dobrana na calibration split
osiągnęła 53,46% Top-1 przy koszcie 85,08%; full-depth na held-out połowie
miał 53,78%. To strata 0,32 pp za około 15% potencjalnie pominiętych bloków.

Wynik jest dodatnim sygnałem, nie dowodem publikacyjnym. Trwa identyczna
replikacja seed 42; seed 2026 jest przygotowany.

## Co nie wyszło

1. Uncertainty routing: uncertainty nie była dobrze związana z błędem.
2. Stage-value regression: nie pokonała utility-only po zbieżności.
3. Zwiększenie częstotliwości targetów: nie poprawiło agreement.
4. Oracle action cloning: nie generalizowało ponad majority baseline.
5. Dynamic channels jako speedup: aktualne gates nie są wykonywalne strukturalnie.

## Potencjał publikacyjny

Channel routing ma niski potencjał jako samodzielna metoda pozytywna. Może być
dobrze udokumentowanym wynikiem negatywnym lub appendixem uzasadniającym pivot.

Adaptive depth ma umiarkowanie wysoki, lecz niezweryfikowany potencjał:

- oracle i held-out confidence policy są dodatnie;
- akcja jest wykonywalna przez pomijanie bloków;
- możliwy jest jasny wykres Pareto accuracy–cost oraz realny latency;
- istnieją uczciwe baselines: full-depth, static-depth i confidence threshold.

Przed publikacją konieczne są trzy seedy, zamrożony dłuższy protokół, osobna
kalibracja per seed, static-depth, risk–coverage/ECE oraz median/P95 latency
dla forwardu faktycznie kończącego się na wyjściu. Potencjalny claim powinien
dotyczyć kalibrowanego adaptive depth i kompromisu latency–risk, nie nowego
modułu attention.

## Zalecany plan

1. Dokończyć seedy 42 i 2026 obecnego pilota.
2. Powtórzyć held-out threshold audit dla każdego seeda i raportować mean±SD.
3. Jeśli strata accuracy pozostanie <=0,5 pp przy koszcie <=85%, zamrozić
   dłuższy Tiny ImageNet protocol i dodać static-depth.
4. Zaimplementować faktyczne przerwanie forward oraz benchmark latency GPU/CPU.
5. Dopiero potem rozważać ImageNet-100 i przegląd literatury pod claim nowości.

## Artefakty

- `COMPREHENSIVE_RESEARCH_SUMMARY_2026-08-21.md` — retrospektywa;
- `EARLY_EXIT_PILOT_REPORT.md` — pierwszy dodatni pilot;
- `ORACLE_AUDIT_REPORT.md` i `ORACLE_ACTION_PREDICTABILITY_REPORT.md`;
- `FAILURE_REPORT_GLOBAL_UCLA_SEED42.md` i `EXPERIMENT_RESULTS.md`;
- commity adaptive-depth: `f8e2d51`, `e545005`, `24f20a1`.
