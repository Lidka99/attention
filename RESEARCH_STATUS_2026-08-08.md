# Status badań — 2026-08-08

## Cel

Badamy attention jako dynamiczny mechanizm kompresji obliczeń CNN: dla każdego
obrazu aktywujemy dokładnie 40 z 64 grup kanałów (62,5%) i rozdzielamy ten
budżet pomiędzy cztery etapy ResNet-50.

Nowa metoda `value-of-compute` uczy policy kontrfaktycznie: przewiduje, w
którym etapie ograniczenie grup kanałów najbardziej podnosi stratę.

## Co zostało odrzucone

Poprzedni global UCLA, oparty o ogólną uncertainty błędu, przegrał z
utility-only na pełnym CIFAR-100 seed 42: 69,82% vs 71,58%. Raport:
`FAILURE_REPORT_GLOBAL_UCLA_SEED42.md`.

## Co poprawiliśmy

- wydzielony split train/validation/test dla nowych pilotów CIFAR-100;
- kontrfaktyczny target stage value-of-compute;
- floor 4 grupy na etap;
- soft quota z exact rounding do 40 grup, zamiast greedy `[4,16,16,4]`;
- kontrola utility-only przy identycznym koszcie;
- hybrydowy Muon/AdamW jako osobna ablacją optymalizatora.

## Wyniki pilotów

| Zbiór / protokół | value-of-compute | utility-only | Wniosek |
|---|---:|---:|---|
| CIFAR-100, 20 epok, floor (wcześniejszy alokator) | 58,15% test | 58,80% test | Value nie wygrywa; wynik diagnostyczny. |
| CIFAR-100, 50 epok, soft quota, Muon | 66,44% wal. | — | Soft quota stabilizuje budżet. |
| CIFAR-100, 50 epok, soft quota, AdamW | 66,34% wal. | — | Muon ma słaby, jednoseedowy sygnał przewagi. |
| Tiny ImageNet, 20 epok, soft quota + Muon | **58,20% wal.** | **56,35% wal.** | Pierwszy dodatni sygnał: +1,85 pp przy identycznym budżecie. |

## Co ten wynik znaczy, a czego nie znaczy

Tiny ImageNet daje pierwszy konkretny powód, aby kontynuować: value-of-compute
wygrywa z utility-only o 1,85 pp przy identycznym budget i optimizerze.

Replikacja screeningowa seed 123 zakończyła się remisem: 56,78% value i
56,78% utility-only. Średnia różnica dwóch seedów wynosi +0,93 pp dla value.
Value nie przegrywa w żadnym seedzie, więc przechodzi do dłuższego protokołu;
nie jest to jeszcze dowód publikacyjny.

Nie jest to jeszcze wynik publikacyjny, ponieważ mamy tylko jeden seed i 20
epok. Nie wolno twierdzić, że metoda jest już lepsza ogólnie ani raportować
realnego speedupu — obecne bramy nie omijają fizycznie konwolucji.


## Decyzja po pierwszej parze 100-epokowej

Protokół 100 epok dla seeda 42 nie potwierdził przewagi końcowej: value-of-compute osiągnął 58,36% walidacji, a utility-only 58,33% (różnica +0,03 pp). Przewaga +1,85 pp z 20 epok była więc najpewniej efektem dynamiki uczenia, a nie potwierdzonym zyskiem po zbieżności. Pierwszy run utility-only został przerwany po 23 epokach; do porównania używamy pełnego rerunu 100 epok.

Następny i jedyny uzasadniony run eskalacyjny to sparowana replikacja 100 epok dla seeda 123, bez dostępu do testu i bez zmiany hiperparametrów:

1. value-of-compute: `tinyimagenet_value_budget_muon_100e_seed123.yaml`;
2. utility-only: `tinyimagenet_utility_floor_muon_100e_seed123.yaml`.

Nie uruchamiamy 200 epok, trzeciego seeda, ConvNeXt ani ImageNet przed zakończeniem tej pary. Jeśli value nie wygra utility-only o co najmniej 0,5 pp w średniej obu seedów, hipotezę o przewadze jakości odrzucamy na tym mechanizmie i przechodzimy wyłącznie do diagnostyki agreement value--counterfactual.

## Wynik końcowy sparowanej replikacji 100 epok (2026-08-19)

| Seed | value-of-compute | utility-only | Różnica value − utility |
|---:|---:|---:|---:|
| 42 | 58,36% | 58,33% | +0,03 pp |
| 123 | 58,74% | 59,33% | −0,59 pp |
| Średnia | 58,55% | 58,83% | −0,28 pp |

**Decyzja go/no-go: NO-GO dla obecnego mechanizmu.** Metoda nie osiągnęła wymaganej przewagi średniej 0,5 pp; w drugim seedzie przegrywa o 0,59 pp. Nie uruchamiać 200 epok, trzeciego seeda, ConvNeXt ani ImageNet z tą wersją.

Następny etap jest wyłącznie diagnostyczny i tani: (1) test overfitu głowy stage-value na zamrożonym batchu kontrfaktycznych targetów, aby potwierdzić przepływ gradientu; (2) 10-epokowy pilot bez testu, zwiększający częstotliwość targetu z co 32 batchy do co batch, bez jednoczesnej zmiany targetu; (3) tylko jeśli agreement przekroczy losowe 25%, oddzielna ablacją wagi value. Zmianę semantyki targetu (pełne porównanie dopuszczalnych transferów zamiast jednego arbitralnego) rozważyć dopiero, gdy te dwa testy wykażą, że problem nie jest wyłącznie zbyt słabym sygnałem uczenia.
