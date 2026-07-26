# attentionv3 — Uncertainty-Calibrated Closed-Loop Attention

## 0. Cel projektu

`attention_v2` bada głównie, **które kanały można wyłączyć**. Wersja `attentionv3` ma odpowiedzieć na trudniejsze pytanie:

> Czy model może sam zdecydować, *ile* i *gdzie* użyć atencji dla konkretnego obrazu, a decyzja ta będzie jednocześnie szybka, trafna i wiarygodna?

Proponuję projekt **UCLA — Uncertainty-Calibrated Layerwise Attention**: zamkniętą pętlę atencji, która dla każdej próbki przewiduje ważność cech oraz niepewność tej decyzji. Obraz łatwy otrzymuje tańszą ścieżkę obliczeń, a obraz trudny lub nietypowy zachowuje więcej kanałów/tokenów i może uruchomić dodatkowe przetwarzanie.

To nie jest obietnica absolutnej nowości — przed publikacją trzeba wykonać pełny przegląd literatury i wyszukiwanie patentowe. Jest to jednak wyraźnie odróżnialna hipoteza względem obecnego `attention_v2`: **dynamiczna alokacja budżetu sterowana niepewnością, z oceną kalibracji i odporności na zmianę rozkładu**, a nie tylko statyczne lub miękkie pruning.

---

## 1. Motywacja i luka badawcza

Wynik samego FLOPs/Top-1 nie mówi, czy system:

- oszczędza obliczenia na łatwych obrazach, ale nie psuje trudnych;
- wie, kiedy jego decyzja o usunięciu kanałów jest ryzykowna;
- zachowuje się bezpiecznie na obrazach innych niż typowe przykłady ImageNet.

Nowsze prace pokazują, że ujednolicenie dynamicznego i statycznego channel pruning jest już aktywnym kierunkiem, a dynamiczne reguły selekcji tokenów są badane osobno. Dlatego wkład `attentionv3` powinien leżeć w **wspólnym, mierzalnym mechanizmie sterowania budżetem + niepewność + wiarygodność decyzji**, a nie w kolejnym module SE.

Punkty odniesienia do dalszego przeglądu:

- [BilevelPruning, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_BilevelPruning_Unified_Dynamic_and_Static_Channel_Pruning_for_Convolutional_Neural_CVPR_2024_paper.html) — dynamiczne i statyczne channel pruning;
- [GlobalPru](https://www.sciencedirect.com/science/article/pii/S0893608023006937) — globalne, zależne od próbki ocenianie ważności kanałów;
- [Patch Selective Transformer](https://www.sciencedirect.com/science/article/abs/pii/S0262885624003445) — dynamiczne wybieranie patchy;
- [ImageNet distribution-shift robustness](https://proceedings.iclr.cc/paper_files/paper/2024/file/1dd85d064697ec1a258cd2b8755b8c6d-Paper-Conference.pdf) — uzasadnienie, by mierzyć odporność poza standardowym ImageNet validation.

---

## 2. Pytanie główne i pytania szczegółowe

**GQ:** Czy niepewność predykowana przez kontroler atencji pozwala uzyskać lepszy kompromis accuracy–real latency niż stały pruning przy tym samym średnim budżecie obliczeniowym?

| ID | Pytanie | Falsyfikowalna hipoteza |
|---|---|---|
| RQ1 | Czy kontroler zależny od próbki zmniejsza średnią latencję bez istotnej straty Top-1? | Przy tym samym Top-1 średni FLOPs i latency spadną względem stałego pruning. |
| RQ2 | Czy sygnał niepewności poprawia decyzje o zachowaniu kanału/tokena? | Pruning „uncertainty-aware” będzie miał mniejszą stratę na trudnych przykładach niż kontroler bez niepewności. |
| RQ3 | Czy jeden kontroler działa w CNN i ViT? | Ta sama zasada budżetowania przeniesie się z ResNet-50 na DeiT-S/ViT-S po zmianie adaptera cech. |
| RQ4 | Czy adaptacja poprawia odporność na shift? | Przy tym samym średnim koszcie model utrzyma większą część accuracy na ImageNet-C/A/R. |
| RQ5 | Czy decyzje atencji są wiarygodne? | Błąd predykcji/odrzucenia będzie korelował z uncertainty; ECE i risk-coverage poprawią się względem baseline'u. |

---

## 3. Proponowana metoda UCLA

### 3.1. Dwa sygnały ważności

W wybranych warstwach model otrzymuje cechy `h_l`. Lekki kontroler wyznacza:

1. **utility** — ile dana grupa kanałów/tokenów pomaga w rozpoznaniu;
2. **uncertainty** — jak bardzo kontroler nie ufa tej ocenie.

Można zacząć od grupowania kanałów (np. grupy po 8/16 kanałów), aby decyzje były strukturalnie wykonalne na GPU.

Schemat:

```text
features h_l
    ├── utility head ─────┐
    ├── uncertainty head ─┼── budget controller ── mask/keep-ratio
    └── feedback z l+1 ──┘
                              ↓
                       następna warstwa
```

Kontroler dostaje globalny limit `B` (np. 50%, 65%, 80% kosztu bazowego) i wybiera maskę. Dla wysokiej niepewności może zwiększyć lokalny budżet, ale musi zapłacić za to w innych warstwach. Dzięki temu testujemy nie tylko „czy coś usunięto”, lecz **czy budżet trafił do właściwych obrazów i miejsc**.

### 3.2. Uczenie

- warm-up pełnego modelu;
- trening utility/uncertainty z distillation od pełnego nauczyciela;
- differentiable top-k lub Gumbel gates;
- kara za FLOPs i odchylenie od zadanego budżetu;
- kalibracja uncertainty na osobnym podzbiorze walidacyjnym;
- hard pruning i fine-tuning;
- pomiar rzeczywistego latency, nie tylko proxy FLOPs.

Proponowana funkcja celu:

```text
L = CE(student, y)
  + λdistill KL(student, teacher)
  + λbudget (cost/B - 1)^2
  + λuncertainty L_Brier
  + λcalibration L_cal
  + λsmooth L_consistency(augmented_view_1, augmented_view_2)
```

W pierwszym prototypie `L_cal` może być zastąpione Brier score, ponieważ jest stabilniejsze podczas treningu niż bezpośrednie minimalizowanie ECE.

### 3.3. Tryb awaryjny dla trudnych obrazów

Jeżeli uncertainty przekroczy próg ustalony wyłącznie na calibration split, model:

- zachowuje dodatkowe grupy cech; albo
- uruchamia drugą, droższą ścieżkę/refinement.

Raportujemy również **coverage**: jaki odsetek obrazów korzysta z taniej ścieżki, oraz **risk-coverage curve**. Nie wolno przedstawiać samego „średniego speedup” bez pokazania kosztu najtrudniejszych przykładów.

---

## 4. Dane i zakres ImageNet

### Główne dane

- ImageNet-1K / ILSVRC2012: trening, validation, dokładnie opisany preprocessing;
- 50k validation: główny test końcowy, użyty jednokrotnie po zamrożeniu protokołu;
- wydzielony calibration split z treningu lub validation, bez mieszania z końcowym raportem.

### Test odporności

Opcjonalny, ale bardzo wartościowy etap: ImageNet-C, ImageNet-A i ImageNet-R. Są to testy poza standardowym rozkładem, więc wynik należy raportować osobno, a nie łączyć ze zwykłym Top-1.

### Kolejność skalowania

1. ImageNet-100 jako szybki pilot mechanizmu i debugowanie;
2. pełny ImageNet-1K na ResNet-50;
3. transfer zasady na DeiT-S/ViT-S, jeśli zasoby pozwolą.

ImageNet-100 służy tylko do selekcji implementacji i hiperparametrów. Wnioski publikacyjne muszą opierać się na ImageNet-1K.

---

## 5. Eksperymenty

### E0 — kontrola jakości

Odtworzyć baseline z `attention_v2` na tej samej architekturze, seedach, preprocessing i checkpointach. Zweryfikować Top-1/Top-5, FLOPs, parametry i real latency.

### E1 — czy adaptacja daje zysk?

Porównać przy tym samym średnim budżecie:

- ResNet-50 bez attention;
- static channel pruning;
- SE/static attention;
- dynamic utility-only;
- UCLA utility + uncertainty.

Budżety: 50%, 65%, 80%, 90% kosztu bazowego. Co najmniej 3 seedy dla ImageNet-100 i 1–3 seedy dla ImageNet-1K zależnie od kosztu.

### E2 — ablation uncertainty

Usunąć kolejno: feedback, uncertainty head, calibration loss, consistency loss i fallback. Ta tabela ma pokazać, który element odpowiada za wynik.

### E3 — quality of the decision

Sprawdzić:

- korelację uncertainty z błędem;
- ECE, NLL i Brier score;
- risk-coverage;
- stratę accuracy według decyli trudności;
- stabilność maski dla dwóch augmentacji tego samego obrazu.

### E4 — odporność na zmianę rozkładu

Na ImageNet-C/A/R porównać accuracy, mCE/odpowiednie metryki oraz średni i 95-percentyl latency. Test: czy model zwiększa budżet wtedy, gdy standardowy kontroler byłby najbardziej podatny na błąd?

### E5 — CNN kontra ViT

Ten sam kontroler w dwóch adapterach:

- kanały/bloki dla ResNet-50;
- tokeny/warstwy dla DeiT-S lub ViT-S.

Nie zakładać z góry, że wynik będzie taki sam — transfer mechanizmu jest osobną hipotezą.

---

## 6. Baselines i uczciwe porównanie

Minimum:

- pełny ResNet-50;
- L1-norm pruning;
- Taylor pruning;
- Network Slimming;
- metoda dynamicznego/static channel pruning z najbliższej literatury;
- obecne static, SE, feedback i Gumbel z `attention_v2`;
- dynamic utility-only jako najważniejszy baseline ablacyjny.

Każdy model powinien mieć ten sam preprocessing, augmentacje, budżet, liczbę epok fine-tuningu i procedurę pomiaru. FLOPs nie zastępują pomiaru latency na konkretnym GPU/CPU.

---

## 7. Metryki i kryteria sukcesu

### Podstawowe

- Top-1 i Top-5;
- średni FLOPs oraz rzeczywisty latency: mediana i P95;
- throughput, peak memory, liczba parametrów i rozmiar checkpointu;
- średni oraz rozkład keep-ratio per warstwa i per obraz.

### Wiarygodność i odporność

- ECE, NLL, Brier score;
- risk-coverage i accuracy na trudnych decylach;
- ImageNet-C/A/R osobno;
- przedziały bootstrap 95% dla accuracy i latency;
- 3 seedy tam, gdzie wynik ma wspierać główną tezę.

### Kryterium go/no-go

Kontynuować pełny ImageNet-1K tylko wtedy, gdy na ImageNet-100 UCLA przy tym samym średnim koszcie osiąga co najmniej jeden z warunków:

1. wyższy Top-1 niż najlepszy baseline przy tej samej latencji; lub
2. co najmniej 20% niższy średni koszt przy stracie Top-1 nie większej niż 0,5 pp; oraz
3. poprawa risk-coverage albo ECE, nie tylko wyniku klasyfikacji.

Jeśli warunki nie są spełnione, projekt pozostaje wartościowym audytem negatywnym i nie należy sztucznie rozszerzać claimu.

---

## 8. Plan implementacyjny w `attentionv3`

```text
attentionv3/
├── PLAN_BADAWCZY.md
├── README.md
├── configs/
│   ├── imagenet100_resnet50.yaml
│   └── imagenet1k_resnet50.yaml
├── src/
│   ├── attention/
│   │   ├── utility_head.py
│   │   ├── uncertainty_head.py
│   │   └── budget_controller.py
│   ├── models/
│   ├── training/
│   └── evaluation/
├── experiments/
│   ├── 00_baseline_reproduction.py
│   ├── 01_imagenet100_pilot.py
│   ├── 02_uncertainty_ablation.py
│   ├── 03_imagenet1k_main.py
│   └── 04_shift_robustness.py
├── tests/
└── results/
```

Najpierw implementować adapter ResNet-50 i kontroler grup kanałów. Token pruning w ViT dopiero po uzyskaniu stabilnego wyniku w CNN; to ogranicza ryzyko i pozwala szybko obalić lub potwierdzić główną hipotezę.

---

## 9. Reprodukowalność i higiena eksperymentów

- zamrożone pliki YAML i wersja kodu dla każdego runu;
- seed, GPU, wersja PyTorch/CUDA i git commit zapisane w JSON;
- osobny calibration split;
- automatyczny test braku overlapu danych;
- raportowanie confidence intervals, nie tylko najlepszego uruchomienia;
- benchmark latency po rozgrzaniu GPU, z określoną liczbą powtórzeń;
- logowanie pełnej polityki budżetu, także dla błędnych predykcji;
- checkpoint nauczyciela i studenta przechowywany razem z konfiguracją.

---

## 10. Harmonogram minimalny

| Okres | Rezultat |
|---|---|
| Tydzień 1 | specyfikacja protokołu, split calibration, testy danych i baseline latency |
| Tygodnie 2–3 | utility-only na ImageNet-100, sanity checks masek |
| Tygodnie 4–5 | uncertainty head, Brier/calibration, E1–E3 |
| Tydzień 6 | go/no-go i decyzja o ImageNet-1K |
| Tygodnie 7–10 | główny eksperyment ImageNet-1K, seedy i ablations |
| Tygodnie 11–12 | ImageNet-C/A/R, analiza błędów, wykresy Pareto i raport |

---

## 11. Najważniejsze ryzyka

1. **Proxy FLOPs nie przekładają się na latency.** Dlatego latency jest metryką główną, a dynamiczne maski powinny działać na grupach wspieranych przez backend.
2. **Uncertainty może być źle skalibrowane.** Stosować osobny calibration split, Brier score i testy reliability diagram.
3. **Fallback może zjeść zysk.** Raportować osobno koszt ścieżki taniej, drogiej i średnią ważoną.
4. **Wynik na ImageNet-100 może nie skalować się do ImageNet-1K.** Nie traktować pilota jako wyniku końcowego.
5. **Nowość może częściowo pokrywać się z pracą opublikowaną w trakcie projektu.** Przed finalnym claimem wykonać systematyczny przegląd i precyzyjnie zdefiniować różnicę: joint utility–uncertainty, budget conservation, calibration i shift robustness.

## Decyzja projektowa

Najpierw budować **jeden dobrze kontrolowany prototyp na ResNet-50**, nie pięć nowych modułów naraz. Jeśli E1–E3 pokażą, że uncertainty rzeczywiście przewiduje ryzyko błędnego pruning, wtedy rozszerzenie na ViT i ImageNet-C/A/R będzie mocną, naturalną historią badawczą.
