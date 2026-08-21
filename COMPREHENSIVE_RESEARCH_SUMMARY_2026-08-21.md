# attentionv3 — pełne podsumowanie badań

Stan na 2026-08-21. Raport rozdziela wyniki potwierdzone, wyniki negatywne,
ich interpretację oraz rekomendowany następny projekt. Obecne maski zerują
aktywacje, lecz nie omijają konwolucji; żaden wynik nie jest claimem realnego
speedupu.

## Cel kampanii

Pytanie było proste: czy przy dokładnie stałym globalnym budżecie 40 z 64 grup
kanałów ResNet-50 polityka zależna od obrazu potrafi przydzielić obliczenia
lepiej niż static pruning i utility-only? Badano kolejno uncertainty, wartość
kontrfaktyczną etapu oraz destylację decyzji label-informed oracle'a.

Każda kontrola zachowywała backbone, budget, optimizer, dane i seed. Dzięki
temu wynik ujemny oznacza słabość decyzji, a nie różnicę kosztu modelu.

## Zbudowana infrastruktura

- ResNet-50 z exact global budget, floor czterech grup na etap i soft quota;
- warianty static, utility-only, UCLA i value-of-compute;
- deterministyczne splity CIFAR-100, adapter Tiny ImageNet oraz DDP;
- kontrfaktyczne targety, oracle audit, 13-akcyjna przestrzeń transferów i
  maskowanie akcji nielegalnych;
- audyty kalibracji, bootstrapowe sparowane porównania i logi per-stage;
- 41 testów jednostkowych — wszystkie przechodzą.

## Wyniki: uncertainty-aware UCLA

W zamrożonym, 200-epokowym protokole CIFAR-100 global UCLA uzyskał 69,82%
Top-1, a global utility-only 71,58%. Sparowana różnica UCLA minus utility-only
to −1,76 pp, z CI 95% [−2,58; −0,93]. Korelacja uncertainty–błąd wyniosła
tylko 0,054. Wniosek jest jednoznaczny: ogólna uncertainty błędu nie była
użytecznym sygnałem redystrybucji budżetu.

Krótsze lokalne runy CIFAR-100 były dodatkowo niestabilne między seedami i nie
pokonały static ani utility-only. Szczegóły znajdują się w
`EXPERIMENT_RESULTS.md` i `FAILURE_REPORT_GLOBAL_UCLA_SEED42.md`.

## Wyniki: value-of-compute

Soft quota naprawiło kolaps greedy alokatora do wzorców typu `[4,16,16,4]` i
utrzymało dokładny budget. Pilot Tiny ImageNet po 20 epokach wyglądał dobrze
tylko dla seeda 42 (+1,85 pp); seed 123 dał remis. Właściwy protokół 100 epok
nie potwierdził przewagi:

| Seed | value-of-compute | utility-only | różnica |
|---:|---:|---:|---:|
| 42 | 58,36% | 58,33% | +0,03 pp |
| 123 | 58,74% | 59,33% | −0,59 pp |
| średnia | 58,55% | 58,83% | −0,28 pp |

Sygnał z 20 epok odzwierciedlał najpewniej dynamikę uczenia, nie jakość po
zbieżności. Z tego powodu nie uruchomiono 200 epok, trzeciego seeda, ConvNeXt
ani ImageNet z tym mechanizmem.

## Co mówi oracle

Oracle zna etykietę tylko podczas audytu i wybiera najlepszy legalny lokalny
transfer czterech grup. Na 512 obrazach Tiny ImageNet, seed 123, ma duży
potencjał:

| Punkt startowy | baseline Top-1 | oracle Top-1 | zysk | redukcja NLL |
|---|---:|---:|---:|---:|
| value-of-compute | 60,16% | 71,09% | +10,94 pp | 1,533 |
| utility-only | 60,74% | 72,66% | +11,91 pp | 1,522 |

Nie jest to wynik inferencyjny, lecz górna granica. Dowodzi, że problemem nie
jest brak potencjału samej przestrzeni transferów, lecz brak obserwowalnej,
generalizującej policy.

## Diagnostyka uczenia policy

Na zamrożonym batchu 64 przykładów głowa policy obniżyła stage-value loss z
1,388 do 1,053 i podniosła agreement z 40,6% do 76,6%. Gradient i pojemność
głowy działają. Natomiast target liczony co batch, zamiast co 32 batche, dał
po 10 epokach agreement 7,19% i loss 1,385. Problemem nie jest wyłącznie
rzadki nadzór, lecz niestacjonarny target i ograniczona informacja wejściowa.

## Oracle-distilled budget routing

Zamiast regresji czterech wartości wprowadzono `noop` plus kierunkowe
transfery, z maską akcji nielegalnych. Held-out predykcja oracle'a nie
generalizuje:

| Cecha i action space | val. accuracy | majority | regret |
|---|---:|---:|---:|
| stem, 13 akcji globalnych | 20,90% | 23,83% | 1,534 NLL |
| layer1, 7 akcji tylko dla etapów 2–4 | 31,64% | 33,98% | 1,410 NLL |
| layer2, 3 akcje tylko dla etapów 3–4 | 59,77% | 61,13% | 0,565 NLL |

Późniejsze cechy zmniejszają regret, ale nawet po layer2 policy nie przebija
prostej decyzji majority (`noop`). Wraz z tym wynikiem zamykamy globalne oraz
etapowe channel-budget routing jako główną tezę. Nie należy dalej stroić tych
samych wag strat, temperatur ani wybierać korzystnego seeda.

## Co nie wyszło

1. Global uncertainty jako sygnał przydziału kanałów.
2. Regresja wartości etapu z pojedynczej kontrfaktycznej ablacją.
3. Naprawa regresji jedynie większą częstotliwością targetów.
4. Destylacja globalnej akcji oracle z cech po stemie.
5. Sekwencyjny router po layer1 i późny router po layer2 jako polityka
   przewyższająca `noop`.

To nie są porażki implementacyjne. Każda ma dopasowaną kontrolę, checkpoint
lub held-out audit. Są wartościowym wynikiem negatywnym, bo zawężają warunki,
w których dynamiczne channel routing nie ma podstaw do skalowania.

## Gdzie pozostaje potencjał

### Adaptive depth / early exit — rekomendowany pivot

Najbardziej uzasadniony nowy temat to decyzja po stage 2 lub 3: zatrzymać
klasyfikację, uruchomić dalsze bloki albo refinement. W przeciwieństwie do
maski kanałów jest to akcja wykonywalna, więc można mierzyć medianę i P95
latency. Ma też bogatsze cechy semantyczne, gdzie regret oracle'a już maleje.

Pierwszy eksperyment nie powinien od razu trenować nowej policy. Należy zrobić
oracle audit stop/refine przy stałym średnim koszcie i porównać go z confidence
threshold oraz static depth. Dopiero wyraźny zysk oracle'a nad tymi kontrolami
uzasadnia trzy seedy Tiny ImageNet i późniejsze ImageNet-100.

### Strukturalne wykonanie

Jeżeli główną wartością ma być szybkość, practicalny projekt to static
structured pruning albo skipowanie bloków residual. Aktualne gates nie dają
speedupu. Dynamiczna decyzja bez wykonywalnej akcji nie powinna być dalej
skalowana.

### Co warto zachować

Exact budget, soft quota, oracle audit, action masking, utility-only control,
splity danych, DDP i audyt bootstrapowy pozostają użytecznymi narzędziami dla
early exit, pruning oraz nowych backbone'ów.

## Zalecana kolejność dalszych prac

1. Zamrozić routing kanałów jako wynik negatywny z pełną dokumentacją.
2. Dodać głowy klasyfikacyjne po stage 2, 3 i 4 ResNet oraz oracle stop/refine.
3. Wykonać krótki CIFAR-100 pilot z static-depth i confidence controls.
4. Po pozytywnej bramce oracle: trzyseadowy Tiny ImageNet oraz realny latency.
5. Dopiero wtedy rozważać ImageNet-100, ConvNeXt i ViT.

## Artefakty

- global UCLA: `FAILURE_REPORT_GLOBAL_UCLA_SEED42.md`;
- pełne wyniki: `EXPERIMENT_RESULTS.md` i `RESEARCH_STATUS_2026-08-08.md`;
- value-of-compute: `VALUE_BUDGET_RESEARCH_LOG.md`;
- oracle: `ORACLE_AUDIT_REPORT.md`;
- policy: `POLICY_DIAGNOSTICS.md` i `ORACLE_ACTION_PREDICTABILITY_REPORT.md`;
- routing sekwencyjny: `SEQUENTIAL_ROUTING_LOG.md` i
  `FUTURE_STAGE_ROUTING_REPORT.md`.
