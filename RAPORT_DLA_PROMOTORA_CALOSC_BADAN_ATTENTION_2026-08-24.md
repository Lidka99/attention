# Sprawozdanie z badań nad attention, channel pruning i adaptive inference

**Autorka projektu:** Lidia Cichońska  
**Data raportu:** 24 sierpnia 2026 r.  
**Zakres:** `attention_v2` oraz `attentionv3`  
**Charakter dokumentu:** kompletne sprawozdanie obejmujące wyniki pozytywne,
negatywne i nierozstrzygnięte, ograniczenia, decyzje go/no-go oraz plan
dalszych prac

---

## Krótkie podsumowanie

Badania nie potwierdziły przewagi dynamicznego channel attention nad prostymi,
dopasowanymi kosztowo kontrolami. UCLA uzyskał 69,82% wobec 71,58% dla
utility-only, value-of-compute po 100 epokach osiągnął średnio 58,55% wobec
58,83%, a training-only semantic feedback pogorszył wynik o 5,36 punktu
procentowego. Wysoka proxy sparsity miękkich bramek nie przełożyła się na
rzeczywiste przyspieszenie, ponieważ konwolucje nadal były wykonywane.

Jednocześnie audyt oracle wykazał potencjał lepszej alokacji kanałów na
poziomie około 11 punktów procentowych, lecz decyzji tej nie udało się
wiarygodnie przewidzieć z cech dostępnych podczas inferencji. Najbardziej
obiecującym kierunkiem okazał się adaptive depth: w trzech seedach zachował
średnią dokładność w granicy −0,07 punktu procentowego względem pełnej sieci
przy koszcie 84,92%, a dla batch size 1 obniżył szacowaną medianę czasu o około
14,6%.

Wynik adaptive depth pozostaje nierozstrzygnięty publikacyjnie: metoda wygrała
ze static-depth w jednym seedzie i minimalnie przegrała w drugim, a trzeci
seed nie został jeszcze wykonany. Najbliższym krokiem jest domknięcie tej
kontroli, a następnie — tylko po pozytywnej bramce — dłuższy trening i pełny
benchmark accuracy–risk–latency.

---

## 1. Przebieg badań i najważniejsze wyniki

Projekt rozpoczął się od badania uczonych bramek kanałowych w CNN. Pierwszym
celem było uzyskanie kompresji przez nadawanie filtrom uczonych wag attention,
a następnie progowanie kanałów o małej wadze. W `attention_v2` zbudowano kilka
wariantów mechanizmu — statyczny, SE, Gumbel i koncepcyjny feedback — oraz
rozbudowany framework diagnostyczny M1–M6. Najbardziej wiarygodny eksperyment
CIFAR-10 wykazał, że wszystkie badane mechanizmy attention osiągały niższą
dokładność od standardowego ResNet-18. Statyczne bramki osiągały wysokie proxy
sparsity, ale wartości bramek były bardzo wrażliwe na próg, a model nadal
wykonywał pełne gęste konwolucje. Nie uzyskano więc rzeczywistego
przyspieszenia. Jednocześnie framework markerów okazał się użyteczny jako
narzędzie ujawniające degradację per klasa, koncentrację błędów, niestabilność
progu, zmianę reprezentacji i ryzyko kwantyzacyjne.

W `attentionv3` pytanie badawcze zaostrzono. Zamiast dowolnej miękkiej
rzadkości wprowadzono dokładny budżet: model ResNet-50 miał zachowywać 40 z 64
grup kanałów i dynamicznie rozdzielać je pomiędzy cztery etapy. Najpierw
sprawdzono uncertainty-aware routing (UCLA), następnie kontrfaktyczne
value-of-compute, a potem destylację decyzji oracle'a. Każdy kolejny etap był
odpowiedzią na zdiagnozowaną słabość poprzedniego.

Najmocniejszy pełny wynik dla UCLA jest negatywny: po 200 epokach na CIFAR-100
UCLA osiągnął 69,82%, a dopasowany kosztowo utility-only 71,58%. Sparowana
różnica wyniosła −1,76 punktu procentowego, z 95% CI [−2,58; −0,93]. Korelacja
predykowanej uncertainty z błędem wyniosła tylko 0,054. Wariant
value-of-compute po 100 epokach na Tiny ImageNet również nie pokonał kontroli:
średnio 58,55% wobec 58,83% dla utility-only na dwóch seedach.

Audyt label-informed oracle'a wykazał jednak bardzo duży teoretyczny potencjał
samej przestrzeni transferów kanałów: +10,94 do +11,91 pp Top-1 na 512
obrazach. Kolejne eksperymenty pokazały, że problemem jest przewidywalność tej
decyzji bez etykiety. Polityki oparte na cechach po stemie, `layer1` i
`layer2` nie przebijały prostego majority baseline na held-out danych.
Dynamiczny channel-budget routing został dlatego zamknięty jako główny
kierunek pozytywnej publikacji.

Najbardziej obiecującym dotychczas pivotem jest adaptive depth / early exit.
W trzech seedach Tiny ImageNet prosta polityka progów confidence zachowała
średnią dokładność w granicy −0,07 pp względem pełnej ścieżki, przy koszcie
około 84,92%. Rzeczywisty prefix-forward na GPU zmniejszył szacowany czas
batch-1 z 5,65 ms do około 4,82 ms, czyli o 14,6%. Wynik nie jest jeszcze
gotowy publikacyjnie, ponieważ kontrola static-depth dała wynik mieszany:
adaptive depth wygrał z nią w seedzie 42, lecz przegrał minimalnie w seedzie
123. Brakuje static-depth seed 2026, dłuższego treningu, pełnej analizy
risk–coverage i benchmarków end-to-end.

Po analizie przeglądu `COMPUTATIONAL_INTELLIGENCE_Cichonska.pdf` wykonano
dodatkowy test semantic-feedback attention używanego wyłącznie podczas
treningu. Hipoteza zakładała, że głębokie cechy pomogą nauczyć statyczną maskę,
która nie potrzebuje feedbacku w inferencji. Wynik był wyraźnie negatywny:
49,83% wobec 55,19% dla identycznej kontroli statycznej, czyli −5,36 pp.
Mechanizm zamknięto bez dalszych seedów.

Najważniejszy wniosek całej kampanii brzmi następująco:

> Uczone bramki kanałowe dobrze opisują ważność i umożliwiają diagnostykę, ale
> badane warianty dynamicznego routingu nie nauczyły się generalizującej
> decyzji lepszej od prostych kontroli. Najbardziej wiarygodny praktyczny
> sygnał dotyczy pomijania całych późnych bloków przez early exit, a nie
> selekcji kanałów wewnątrz już wykonywanych konwolucji.

---

## 2. Główne pytania badawcze

Badania w obu repozytoriach odpowiadały kolejno na sześć pytań:

1. Czy uczona bramka kanałowa może wskazać kanały zbędne bez dużej utraty
   jakości klasyfikacji?
2. Czy raportowane sparsity oznacza rzeczywistą redukcję pamięci, FLOPs lub
   latency?
3. Czy dynamiczne attention zależne od obrazu może lepiej wykorzystywać stały
   budżet niż jedna maska statyczna?
4. Czy uncertainty albo kontrfaktyczna wartość obliczenia jest dobrym sygnałem
   decyzji budżetowej?
5. Czy istnieje użyteczna decyzja oracle i czy da się ją przewidzieć z cech
   dostępnych podczas inferencji?
6. Jeżeli routing kanałów nie działa, czy adaptacyjna głębokość daje lepszy,
   sprzętowo wykonywalny kompromis accuracy–latency?

Ważną zasadą interpretacyjną jest rozróżnienie:

- **soft gating / masking** — konwolucja zostaje wykonana, a wynik jest dopiero
  skalowany albo zerowany;
- **structural pruning** — kanał i związane z nim parametry są fizycznie
  usunięte z modelu;
- **conditional execution** — operator lub cały blok nie jest uruchamiany dla
  części wejść;
- **proxy cost** — teoretyczny udział zachowanych grup lub bloków;
- **real latency** — rzeczywiście zmierzony czas wykonania na określonym
  urządzeniu.

To rozróżnienie zmieniło kierunek projektu. Duża liczba małych gate'ów nie
jest sama w sobie dowodem kompresji, jeżeli graf obliczeniowy pozostaje gęsty.

---

## 3. Chronologia badań i decyzji

| Etap | Główna hipoteza | Najważniejszy wynik | Decyzja |
|---|---|---|---|
| `attention_v2`: static/SE/Gumbel | uczone bramki poprawią kompromis accuracy–sparsity | wysoka proxy sparsity, ale gorsza accuracy i brak realnego speedupu | zachować framework, wzmocnić protokół |
| `attention_v2`: markery M1–M6 | sama accuracy nie wystarcza do oceny kompresji | markery ujawniły niestabilność progu i różnice per klasa | wynik metodologicznie użyteczny |
| `attention_v2`: CIFAR-100 | statyczne attention skaluje się do trudniejszego zbioru | β=0,5: 71,59% vs 71,83% baseline przy 45,5% proxy pruning | obiecujący pilot, ale tylko jeden seed i bez fizycznego pruning |
| `attentionv3`: lokalne UCLA | uncertainty poprawi dynamiczną selekcję kanałów | średnio 51,02%, poniżej static 51,80% | brak przewagi, potrzebny pełny test |
| `attentionv3`: global UCLA 200 epok | uncertainty lepiej rozdzieli dokładny budżet 40/64 | 69,82% vs 71,58%; delta −1,76 pp | no-go |
| value-of-compute | kontrfaktyczny target nauczy wartości budżetu per etap | 58,55% vs 58,83% po 100 epokach | zamknięcie regresji value |
| oracle transfer | przestrzeń akcji może zawierać lepsze decyzje | +10,94 do +11,91 pp oracle gain | badać przewidywalność |
| oracle action cloning | policy nauczy się akcji oracle | held-out poniżej majority po stem/layer1/layer2 | zamknąć channel routing |
| adaptive depth | łatwe próbki mogą kończyć wcześniej | −0,07 pp przy koszcie 84,92%; realny sygnał latency | kontynuować po kontrolach |
| static-depth | adaptive exit musi wygrać z prostym stałym skróceniem | seed 42 dodatni, seed 123 ujemny | wynik mieszany, potrzebny seed 2026 |
| training-only feedback | głęboka semantyka nauczy lepszą maskę statyczną | 49,83% vs 55,19%, delta −5,36 pp | zamknąć mechanizm |

---

## 4. Etap `attention_v2`: learnable filter attention

### 4.1. Cel pierwszej wersji projektu

`attention_v2` badał, czy ważność filtrów może być uczona równocześnie z
klasyfikacją. Zamiast usuwać filtry jedynie na podstawie norm wag po treningu,
model otrzymywał bramki mnożące aktywacje kanałów. Mała wartość bramki miała
oznaczać, że kanał jest kandydatem do usunięcia.

Rozważano następujące warianty:

- **SCA / static L1:** jedna uczona wartość na kanał, wspólna dla wszystkich
  obrazów, z karą L1 sterowaną współczynnikiem β;
- **SE:** input-dependent channel scaling przez squeeze-and-excitation;
- **Gumbel:** różniczkowalna aproksymacja dyskretnego wyboru kanałów;
- **BGA / feedback attention:** koncepcja sterowania wcześniejszymi bramkami
  informacją z późniejszej części modelu;
- **ChannelGate:** miękkie bramki sigmoid/softplus z karą średniej aktywacji.

Repozytorium zawierało także eksploracyjne implementacje dodatkowych
regularizatorów, knowledge distillation, iterative magnitude pruning,
sparse-training/RigL, dynamicznego β, dynamic gating, wariantu ViT i narzędzi
Pareto. Nie wszystkie te linie mają zachowany jednolity finalny protokół i
pełny audyt. W niniejszym raporcie są traktowane jako prace implementacyjne,
a nie jako potwierdzone wyniki naukowe.

### 4.2. Audyt i uporządkowanie kodu

W trakcie przeglądu `attention_v2` wykryto kilka problemów, które mogłyby
prowadzić do zbyt silnych wniosków:

- pierwotna strata Gumbel działała przeciwnie do zamierzonego celu sparsity;
- część starszych wyników pochodziła z niespójnych skryptów;
- standardowy checkpoint ResNet-18 był wcześniej oceniany przez nie w pełni
  zgodną ścieżkę modelu;
- małe gate'y były miejscami interpretowane jako speedup, mimo że konwolucje
  nadal były wykonywane;
- jeden seed był niewystarczający do twierdzenia o przewadze metody.

Błąd Gumbel poprawiono i dodano test regresyjny. Uporządkowano reprodukcję,
pełny test CIFAR-10, bootstrap, porównania sparowane, McNemara, analizę per
klasa, sweep progu, FLOPs i latency. Starsze nieporównywalne artefakty nie są
używane jako dowód.

### 4.3. CIFAR-10 — pełny audyt checkpointów

Protokół obejmował ResNet-18, seed 42 i pełne 10 000 obrazów testowych.

| Model | Top-1 | Bootstrap 95% CI | Delta do baseline | Proxy sparsity przy τ=0,001 |
|---|---:|---:|---:|---:|
| standard ResNet-18 | **93,48%** | 92,98–93,91% | — | nie dotyczy |
| static L1, β=0,2 | 91,58% | 91,06–92,08% | −1,90 pp | 84,3% |
| static L1, β=0,5 | 91,28% | 90,72–91,80% | −2,20 pp | 90,2% |
| SE, β=0,2 | 91,49% | 90,97–92,02% | −1,99 pp | brak strukturalnego pruning |
| Gumbel, β=0,2 | 92,60% | 92,10–93,09% | −0,88 pp | 0% |

Bootstrapowe przedziały dla sparowanej różnicy wobec baseline'u były ujemne:

- static L1 β=0,2: [−2,43; −1,42] pp;
- static L1 β=0,5: [−2,70; −1,73] pp;
- SE β=0,2: [−2,45; −1,49] pp;
- Gumbel β=0,2: [−1,35; −0,43] pp.

W tym seedzie żaden mechanizm nie poprawił dokładności. Gumbel zachował
najwięcej jakości, ale nie usunął żadnych kanałów przy badanych progach.
Static L1 dawał bardzo wysokie proxy sparsity, lecz tracił 1,9–2,2 pp.

### 4.4. Latency i dlaczego proxy sparsity nie było speedupem

Źródłowy audyt raportował następującą batch latency:

| Model | Batch latency | Obrazy/s |
|---|---:|---:|
| standard | 12,19 ms | 10 502 |
| static L1 β=0,2 | 12,69 ms | 10 088 |
| static L1 β=0,5 | 12,71 ms | 10 074 |
| SE β=0,2 | 13,17 ms | 9 718 |
| Gumbel β=0,2 | 12,76 ms | 10 030 |

Modele bramkowane nie były szybsze od baseline'u. To wynik zgodny z
implementacją: bramka działała po gęstej konwolucji. Liczba parametrów oraz
gęste FLOPs także pozostawały praktycznie takie same. Wniosek wdrożeniowy jest
negatywny, ale ważny: soft attention bez przebudowy grafu nie jest metodą
przyspieszania inferencji.

### 4.5. Markery wiarygodności M1–M6

Zbudowano sześć markerów uzupełniających globalną accuracy:

| Marker | Pytanie diagnostyczne |
|---|---|
| M1 | Czy kompresja degraduje wybrane klasy bardziej niż inne? |
| M2 | Czy błędy koncentrują się nieproporcjonalnie w części klas? |
| M3 | Czy pruning jest stabilny na zmianę progu gate'ów? |
| M4 | Czy dynamiczny koszt zależy od klasy lub częstości klasy? |
| M5 | Jak bardzo zmieniają się wewnętrzne reprezentacje? |
| M6 | Czy aktywacje są stabilne w symulacji FP32→INT8? |

W audycie markerów dla statycznych modeli CIFAR-10 uzyskano:

| Checkpoint | M1 accuracy | M2 bias index | M3 fragile ratio | M5 cosine |
|---|---:|---:|---:|---:|
| β=0,0 | 88,51% | 0,4165 | 0,0% | brak baseline'u |
| β=0,2 | 88,90% | 0,4126 | 99,9% | 0,9612 |
| β=0,5 | 88,10% | 0,4084 | 100,0% | 0,9558 |

Te checkpointy markera pochodzą z osobnej ścieżki niż publikacyjny audyt
93,48%, dlatego nie należy mieszać wartości accuracy między tabelami. Ich
wartość leży w diagnostyce względnej. Najważniejszy sygnał to M3: niemal
wszystkie niezerowe gate'y modeli β=0,2/0,5 znajdowały się w obszarze
wrażliwym na próg. Przy τ=0,001 model wyglądał na bardzo rzadki, lecz przy
τ=0,01 wszystkie gate'y wpadały poniżej progu. Oznacza to brak stabilnej,
naturalnej granicy pomiędzy kanałami ważnymi i zbędnymi.

M4 był poprawnie oznaczony jako nieaplikowalny dla maski statycznej. M5
wskazywał umiarkowanie wysokie podobieństwo reprezentacji, ale pewien drift.
M6 był jedynie symulacją kwantyzacji, nie dowodem działania pełnego PTQ/QAT.

### 4.6. Audyt danych

Audyt SHA-256 surowych obrazów wykazał:

| Właściwość | CIFAR-10 | lokalny CIFAR-100 |
|---|---:|---:|
| train / test | 50 000 / 10 000 | 50 000 / 10 000 |
| duplikaty wewnątrz train | 0 | 14 |
| duplikaty wewnątrz test | 0 | 2 |
| dokładny overlap train–test | 0 | 10 |

CIFAR-10 był technicznie spójny. W lokalnej kopii CIFAR-100 znaleziono 10
identycznych obrazów między train i test. To niewielka liczba, która nie
unieważnia automatycznie wyników, ale wymaga deduplikowanej analizy wrażliwości
przed interpretowaniem różnic rzędu kilku dziesiątych punktu procentowego.

Oba zbiory są zbalansowanymi benchmarkami 32×32. Są odpowiednie do kontroli
implementacji, lecz nie reprezentują obrazów wysokiej rozdzielczości,
niezbalansowanych danych produkcyjnych ani zmiany domeny.

### 4.7. CIFAR-100 — statyczne attention po 200 epokach

Źródłowe JSON-y zawierają pełne runy seed 42:

| Wariant | Top-1 | Gate'y poniżej progu | Proxy sparsity |
|---|---:|---:|---:|
| static L1 β=0,0, 4 GPU | 71,83% | 0/1920 | 0% |
| static L1 β=0,2, 4 GPU | 71,25% | 674/1920 | 35,1% |
| static L1 β=0,5, 4 GPU | 71,59% | 874/1920 | 45,5% |

Istnieje również wcześniejszy osobny baseline β=0,0 o wyniku 72,20%. Nie
powinien być mieszany z tabelą 4-GPU jako identyczna kontrola, ponieważ
pochodzi z wcześniejszego przebiegu.

Wariant β=0,5 stracił tylko 0,24 pp wobec dopasowanego baseline'u 71,83%, przy
45,5% gate'ów poniżej progu. Jest to jeden z bardziej obiecujących sygnałów
`attention_v2`, ale pozostaje wynikiem jednoseedowym, zależnym od arbitralnego
progu i bez fizycznego usunięcia kanałów. Nie można go nazywać 45,5%
przyspieszeniem ani 45,5% redukcją FLOPs.

### 4.8. Co rzeczywiście wniósł `attention_v2`

Wynikiem pozytywnym nie jest przewaga nowej architektury. Wartość pierwszej
wersji projektu polega na:

- stworzeniu różnorodnych mechanizmów attention i wspólnego pipeline'u;
- ujawnieniu różnicy między proxy sparsity a realnym kosztem;
- opracowaniu audytowalnych markerów M1–M6;
- wprowadzeniu bootstrapu, testów sparowanych i analizy per klasa;
- wykryciu niestabilności thresholdingu;
- audycie integralności danych;
- przygotowaniu metodologicznej podstawy dla bardziej rygorystycznego
  `attentionv3`.

---

## 5. Etap `attentionv3`: dokładny budżet i uncertainty-aware routing

### 5.1. Dlaczego powstało `attentionv3`

`attention_v2` nie odpowiadał jednoznacznie, ile obliczeń rzeczywiście
przydziela model i czy porównywane warianty mają ten sam koszt. `attentionv3`
wprowadził kontrolę budżetu: cztery etapy ResNet-50, po 16 grup kanałów, z
dokładnie 40 aktywnymi grupami z 64, czyli 62,5%.

Hipoteza UCLA (Uncertainty-Calibrated Layerwise Attention) zakładała, że:

- utility wybiera najbardziej wartościowe grupy;
- uncertainty wskazuje, gdzie decyzja o redukcji jest ryzykowna;
- jeden globalny budżet jest rozdzielany pomiędzy etapy zależnie od obrazu;
- trudniejsze próbki powinny zachowywać obliczenia w bardziej potrzebnych
  miejscach.

### 5.2. Krótkie eksperymenty CIFAR-100, trzy seedy

Przy dopasowanym keep-ratio około 62,5%:

| Wariant | Top-1, średnia ± SD | Keep-ratio |
|---|---:|---:|
| UCLA | 51,02% ± 3,47 pp | 62,45% ± 0,36 pp |
| utility-only | 51,51% ± 0,92 pp | 62,50% |
| static | **51,80% ± 1,51 pp** | 62,50% |

UCLA był średnio o 0,49 pp gorszy od utility-only i o 0,78 pp gorszy od
static. Dodatkowo miał znacznie większą zmienność. Wyniki per seed były
niestabilne: UCLA wygrywał w seedzie 123, ale wyraźnie przegrywał w seedzie
2026. Krótkie runy nie dawały podstaw do claimu, lecz uzasadniły jeden pełny,
zamrożony test mechanizmu.

### 5.3. Global UCLA — zamrożony protokół 200 epok

Protokół obejmował 20 epok warm-up, 120 epok sparsification i 60 epok
fine-tuningu. Porównano checkpointy końcowe po ustalonych 200 epokach.

| Wariant | Top-1 | Bootstrap 95% CI | Keep-ratio |
|---|---:|---:|---:|
| global UCLA | 69,82% | [68,95%; 70,70%] | 62,50% |
| global utility-only | **71,58%** | [70,66%; 72,50%] | 62,50% |

Sparowana różnica `UCLA − utility-only` wyniosła **−1,76 pp**, z 95% CI
**[−2,58; −0,93] pp**. Przedział nie obejmuje zera. Utility-only było
poprawne przy błędzie UCLA dla 1006 obrazów; odwrotna sytuacja wystąpiła dla
830 obrazów.

Diagnostyka UCLA:

| Metryka | Wynik |
|---|---:|
| Top-5 | 90,22% |
| NLL | 2,4737 |
| ECE | 22,50% |
| Brier wieloklasowy | 0,5075 |
| uncertainty–błąd, Pearson | **0,054** |
| średnia uncertainty: trafienie / błąd | 0,00260 / 0,00280 |

Z góry ustalone kryteria wymagały straty accuracy nie większej niż 0,5 pp i
korelacji uncertainty–błąd co najmniej 0,25. Oba warunki zostały wyraźnie
niespełnione.

### 5.4. Dlaczego UCLA nie zadziałał

Analiza kodu i logów wskazała trzy główne przyczyny:

1. **Zapadający się target uncertainty.** Target `hard_error` był liczony z
   błędu modelu na bieżącym minibatchu. Pod koniec treningu accuracy treningowa
   była niemal 100%, więc target prawie zawsze wynosił zero. Głowa nauczyła się
   przewidywać wartość bliską zeru, a nie generalizujące ryzyko.
2. **Brak gradientu klasyfikacyjnego przez decyzję o liczbie grup per etap.**
   Straight-through działał dla wyboru grup wewnątrz etapu, ale twardy `argmax`
   rozdzielający budżet między etapy pozostawał nieróżniczkowalny.
3. **Niedopasowanie celu do decyzji.** Jedna binarna etykieta „obraz poprawny
   lub błędny” nie mówi, który konkretny etap powinien otrzymać dodatkową grupę.

To nie była awaria ResNet ani DDP. Oba modele ukończyły pełne 200 epok i miały
identyczny koszt. Odrzucona została konkretna semantyka uncertainty jako
sygnału alokacji.

---

## 6. Kontrfaktyczne stage value-of-compute

### 6.1. Nowa hipoteza

Po niepowodzeniu ogólnej uncertainty postawiono bardziej bezpośrednie pytanie:

> W którym etapie dodatkowa grupa kanałów najbardziej obniżyłaby stratę dla
> danego obrazu?

Target tworzono kontrfaktycznie, ograniczając kolejne etapy i mierząc zmianę
NLL. Wprowadzono deterministyczny split train/validation/test, floor czterech
grup na etap oraz dokładny globalny budżet.

### 6.2. Problemy wykryte w pilotach

Pierwszy greedy allocator zapadał się do skrajnych wzorców, np.
`[16, 13, 10, 1]`, a po dodaniu floor do `[4, 16, 16, 4]`. Nawet małe różnice
score powodowały nasycenie jednego etapu do maksimum. Wprowadzono soft quota z
exact rounding, która naprawiła tę wadę techniczną i zachowała dokładny budżet.

Pilot diagnostyczny CIFAR-100 osiągnął 60,68% validation; entropy targetu
0,288 wskazywała, że target nie jest trywialny, ale agreement policy z targetem
wyniósł tylko 14,8%. To skierowało eksperymenty na Tiny ImageNet.

### 6.3. Tiny ImageNet: pilot 20 epok i właściwy test 100 epok

W 20-epokowym pilocie seed 42 value-of-compute wyglądał obiecująco, z przewagą
około +1,85 pp. Seed 123 dał remis. Po 100 epokach wczesny sygnał zniknął:

| Seed | Value-of-compute | Utility-only | Różnica |
|---:|---:|---:|---:|
| 42 | 58,36% | 58,33% | +0,03 pp |
| 123 | 58,74% | 59,33% | −0,59 pp |
| średnia | 58,55% | **58,83%** | −0,28 pp |

Wniosek metodologiczny jest istotny: 20 epok było właściwym screeningiem, ale
nie wystarczało do oceny po zbieżności. Dłuższy protokół pokazał, że dodatni
pilot odzwierciedlał dynamikę uczenia, a nie trwałą przewagę policy. Dlatego
nie uruchomiono kosztownego 200-epokowego rozszerzenia ani ImageNet dla tej
metody.

### 6.4. Czy problemem była pojemność głowy?

Na zamrożonych 64 obrazach i stałych targetach głowa policy obniżyła loss z
1,388 do 1,053 i zwiększyła agreement z 40,6% do 76,6%. Gradient i pojemność
były więc wystarczające do zapamiętania targetu.

Jednocześnie eksperyment z targetem liczonym co batch zamiast co 32 batche dał
po 10 epokach tylko 7,19% agreement i loss 1,385. Sama większa częstotliwość
nadzoru nie rozwiązała problemu. Główna trudność była związana z
niestacjonarnością targetów oraz niewystarczającą obserwowalnością przyszłej
wartości z wczesnych cech.

---

## 7. Oracle transferów i granica przewidywalności

### 7.1. Czy przestrzeń akcji zawiera lepsze rozwiązania?

Label-informed oracle znał etykietę wyłącznie podczas audytu. Dla każdego z
512 obrazów sprawdzał bieżącą alokację oraz legalne transfery czterech grup
między etapami i wybierał wariant o najniższym NLL.

| Checkpoint startowy | Baseline Top-1 | Oracle Top-1 | Zysk | Redukcja NLL |
|---|---:|---:|---:|---:|
| value-of-compute | 60,16% | 71,09% | +10,94 pp | 1,533 |
| utility-only | 60,74% | 72,66% | +11,91 pp | 1,522 |

Oracle zmieniał decyzję dla 76–81% obrazów. Wynik nie jest osiągalną metodą
inferencyjną, ponieważ korzysta z prawdziwej etykiety. Pokazuje natomiast, że
legalne transfery mają duży potencjał. Porażka value-of-compute nie wynikała z
tego, że wszystkie alokacje są równoważne.

### 7.2. Destylacja jawnej akcji oracle

Zdefiniowano 13 akcji: `noop` i 12 kierunkowych transferów donor→recipient.
Następnie sprawdzano, czy prosty klasyfikator potrafi przewidywać akcję oracle
na held-out danych.

| Cechy dostępne policy | Action space | Held-out accuracy | Majority | Mean regret |
|---|---|---:|---:|---:|
| stem | 13 akcji globalnych | 20,90% | 23,83% | 1,534 NLL |
| po `layer1` | 7 akcji dla etapów 2–4 | 31,64% | 33,98% | 1,410 NLL |
| po `layer2` | 3 akcje dla etapów 3–4 | 59,77% | 61,13% | 0,565 NLL |

Im późniejsze cechy, tym mniejszy regret, ale żadna policy nie przebiła
większościowej decyzji `noop`. Wczesne głowy potrafiły częściowo zapamiętać
train, ale nie generalizowały. Po `layer2` przestrzeń akcji była już mała, a
mimo to prosty majority baseline pozostawał lepszy.

### 7.3. Decyzja o zamknięciu channel-budget routing

Po serii UCLA → value-of-compute → fixed-target overfit → dense targets →
oracle actions → routing sekwencyjny dalsze strojenie tych samych temperatur,
wag strat lub seedów nie było uzasadnione. Zamknięto jako główną tezę:

- globalny uncertainty-aware channel routing;
- kontrfaktyczną regresję wartości etapów;
- globalne klonowanie akcji oracle;
- routing przyszłych etapów po `layer1` i `layer2`.

Pozostaje wartościowy wynik negatywny: bardzo dobry label-informed oracle nie
implikuje, że jego decyzja jest przewidywalna z obserwacji dostępnych podczas
inferencji.

---

## 8. Pivot do adaptive depth / early exit

### 8.1. Uzasadnienie

W channel gatingu nawet prawidłowa maska nie omijała konwolucji. Early exit ma
inną własność: jeżeli próbka kończy po stage 2 lub stage 3, późniejsze bloki
nie muszą być wykonane. Akcja jest bezpośrednio realizowalna, a decyzja może
korzystać z bogatszych cech semantycznych.

ResNet-50 otrzymał klasyfikatory po stage 2, stage 3 i stage 4. Użyto wspólnej
straty `0,25·CE(stage2) + 0,5·CE(stage3) + CE(stage4)` i 20-epokowego pilota
Tiny ImageNet.

### 8.2. Jakość wyjść i oracle early exit

| Seed | Stage 2 | Stage 3 | Stage 4 | Oracle Top-1 | Oracle cost |
|---:|---:|---:|---:|---:|---:|
| 42 | 37,46% | 53,07% | 55,14% | 63,62% | 75,21% |
| 123 | 35,67% | 52,53% | 54,22% | 62,53% | 76,05% |
| 2026 | 37,24% | 51,57% | 53,56% | 61,90% | 75,54% |

Oracle wybierał najwcześniejsze poprawne wyjście. Jest label-informed górną
granicą, nie polityką wdrożeniową. W każdym seedzie był wyraźnie lepszy od
stage 4 przy koszcie około 75–76%, więc przestrzeń decyzji stop/refine ma
potencjał.

### 8.3. Confidence policy z niezależną kalibracją

Walidację podzielono deterministycznie: pierwsze 5000 obrazów służyło do
doboru progów, drugie 5000 do held-out oceny. Cel średniego kosztu wynosił
maksymalnie 85% pełnej głębokości.

| Seed | Policy Top-1 | Full-depth Top-1 | Różnica | Cost |
|---:|---:|---:|---:|---:|
| 42 | 54,96% | 54,96% | 0,00 pp | 84,75% |
| 123 | 53,46% | 53,78% | −0,32 pp | 85,08% |
| 2026 | 54,12% | 54,02% | +0,10 pp | 84,95% |
| średnia ± SD | 54,18% ± 0,75 | 54,25% ± 0,62 | −0,07 pp ± 0,21 | 84,92% ± 0,17 |

To pierwszy stabilny, wieloseedowy dodatni sygnał w `attentionv3`. Prosta
policy zachowała praktycznie całą jakość stage 4 przy około 15% mniejszym
proxy koszcie.

### 8.4. Rzeczywisty prefix latency

Zaimplementowano forward kończący wykonanie dokładnie na wybranym wyjściu.
Benchmark CUDA, batch size 1, checkpoint seed 42, 20 warm-up i 60 pomiarów:

| Wyjście | Mediana | P95 |
|---|---:|---:|
| stage 2 | 2,63 ms | 2,94 ms |
| stage 3 | 4,62 ms | 5,06 ms |
| stage 4 | 5,65 ms | 5,79 ms |

Dla rozkładu wyjść policy seed 42 ważona oczekiwana mediana wynosi około
4,82 ms, czyli około 14,6% mniej niż pełne 5,65 ms. W odróżnieniu od channel
masking jest to realny pomiar pominiętych bloków.

Nie jest to jeszcze pełny benchmark systemowy. Brakuje kosztu samej decyzji,
end-to-end routingu, batchowania dynamicznego, pomiaru CPU, wielu urządzeń i
powtórzenia wszystkich seedów. W praktycznym batchu wolna próbka może
zatrzymać cały batch, dlatego batch-1 prefix latency nie wystarcza do claimu
produkcyjnego.

### 8.5. Najważniejsza kontrola: static depth

Dynamiczny model należy porównać z siecią zawsze kończącą po stage 3.

| Seed | Static stage 3 | Adaptive policy | Adaptive full-depth | Wniosek |
|---:|---:|---:|---:|---|
| 42 | 53,55% | 54,96% | 54,96% | adaptive +1,41 pp kosztem +3,5 pp cost |
| 123 | 53,67% | 53,46% | 53,78% | adaptive −0,21 pp wobec static |
| 2026 | brak | 54,12% | 54,02% | kontrola niewykonana |

Seed 42 sugerował korzystny punkt Pareto. Seed 123 nie potwierdził kierunku
różnicy. Nie wolno więc obecnie twierdzić, że adaptive depth dominuje dobrze
wytrenowaną sieć o stałej głębokości. Seed 2026 i średnia ± SD są warunkiem
minimalnym dalszej interpretacji.

### 8.6. Poziom dojrzałości wyniku early exit

Early exit jest **obiecującym pilotem**, nie gotową nowością publikacyjną.
Sama idea early exit jest dobrze znana. Także kalibracja niepewności i
spójności między wyjściami jest aktywnie badana, m.in. w pracy
[Early-Exit Neural Networks with Nested Prediction Sets](https://proceedings.mlr.press/v244/jazbec24a.html),
a problem overconfidence w sieciach dynamicznych analizuje
[Fixing Overconfidence in Dynamic Neural Networks](https://openaccess.thecvf.com/content/WACV2024/html/Meronen_Fixing_Overconfidence_in_Dynamic_Neural_Networks_WACV_2024_paper.html).

Potencjalny wkład nie może brzmieć „dodaliśmy early exit do ResNet”. Musiałby
dotyczyć rygorystycznej kontroli ryzyka, latency albo zachowania dynamicznych
batchy, popartego lepszym protokołem niż obecny pilot.

---

## 9. Powrót do attention: training-only semantic feedback

### 9.1. Motywacja z przeglądu literatury

Lokalny dokument `COMPUTATIONAL_INTELLIGENCE_Cichonska.pdf` wskazywał jako
potencjalną lukę połączenie:

- task-aware, input-dependent channel attention;
- informacji semantycznej z warstw głębokich do wcześniejszych gate'ów;
- automatycznego ustalenia liczby zachowanych kanałów;
- trwałego, strukturalnego usunięcia kanałów i realnego latency.

Luka wymaga ostrożności. Łączenie dynamicznego i statycznego pruning jest już
przedmiotem prac takich jak
[BilevelPruning](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_BilevelPruning_Unified_Dynamic_and_Static_Channel_Pruning_for_Convolutional_Neural_CVPR_2024_paper.html),
a selektywne przetwarzanie kanałów przy statycznym grafie występuje w
[CNN Mixture-of-Depths](https://openaccess.thecvf.com/content/ACCV2024/html/Cakaj_CNN_Mixture-of-Depths_ACCV_2024_paper.html).
Sam feedback albo połączenie static/dynamic nie jest więc bezpiecznym claimem
nowości.

### 9.2. Testowana hipoteza

Aby uniknąć kosztu dynamicznej ścieżki w inferencji, zaprojektowano mechanizm,
w którym stage-4 semantic feedback jest dostępny wyłącznie podczas treningu.
Feedback korygował wybór 10 z 16 grup stage 1. Walidacja i deployment używały
tylko jednej statycznej maski; nie uruchamiały drugiego przebiegu ani głowy
feedback.

### 9.3. Kontrola statyczna i wynik feedback

Matched protocol: Tiny ImageNet, seed 42, ResNet-50, 20 epok, AdamW, batch 64
na GPU, 4 GPU, 10/16 grup stage 1.

| Ramię | Ostatnia deployment Top-1 | Najlepsza Top-1 | Delta ostatnia |
|---|---:|---:|---:|
| static channel selection | **55,19%** | **56,26%** | — |
| training-only semantic feedback | 49,83% | 52,88% | **−5,36 pp** |

Feedback score nie zanikł podczas treningu. Mechanizm rzeczywiście zmieniał
decyzje, ale decyzje zależne od obrazu znikały podczas deploymentu. Najbardziej
prawdopodobną przyczyną jest train–deployment mismatch: klasyfikator uczył się
z maskami korygowanymi przez semantykę próbki, natomiast w walidacji musiał
działać z jednym stałym rankingiem.

Różnica −5,36 pp jest wystarczająco duża, aby zamknąć tę implementację bez
wydawania zasobów na seedy 123 i 2026. Nowy seed nie naprawiłby strukturalnego
niedopasowania procedury.

### 9.4. Błędy wykonawcze oddzielone od wyniku naukowego

Przed poprawnym runem wystąpiły trzy problemy techniczne:

- brak `PYTHONPATH=src` pod `torchrun`;
- OOM dla batch 128/GPU;
- błąd DDP, ponieważ `feedback_head` jest celowo nieużywany w ramieniu static.

Poprawiono ścieżkę importu, zamrożono batch 64/GPU i włączono obsługę unused
parameters. Końcowe oba runy ukończyły 20 epok. Wynik −5,36 pp nie jest więc
skutkiem przerwanego treningu ani OOM.

---

## 10. Synteza wyników pozytywnych

### 10.1. Wyniki metodologiczne i infrastrukturalne

Najsilniejszym osiągnięciem projektu jest coraz bardziej rygorystyczny sposób
falsyfikowania hipotez:

- wspólny budżet i matched controls;
- deterministyczne splity danych;
- rozdzielenie validation i test;
- bootstrap i sparowane różnice;
- utility-only, static i full-depth jako obowiązkowe kontrole;
- dokładny budżet 40/64, floor i soft quota;
- kontrfaktyczne targety i label-informed oracle audits;
- legal action masking i held-out predictability;
- forward rzeczywiście kończący wykonanie dla early exit;
- median/P95 latency zamiast samego FLOPs;
- jawne kryteria go/no-go;
- zapisywanie również wyników negatywnych.

### 10.2. Konkretne pozytywne obserwacje

1. Static L1 na CIFAR-100 β=0,5 zachował 71,59% wobec 71,83% baseline przy
   45,5% gate'ów poniżej progu. To wskazuje, że redundancja kanałów istnieje,
   choć nie dowodzi realnej kompresji.
2. Oracle transferów kanałów wykazał bardzo duży potencjał label-informed,
   +10,94 do +11,91 pp. Przestrzeń akcji nie była trywialna.
3. Policy potrafiła overfitować zamrożony target, więc podstawowy przepływ
   gradientu i pojemność głowy działały.
4. Oracle early exit poprawiał Top-1 przy koszcie około 75–76%, co potwierdza
   istnienie użytecznej przestrzeni stop/refine.
5. Confidence early exit replikował się w trzech seedach: −0,07 pp średniej
   różnicy przy koszcie 84,92%.
6. Prefix-forward dał mierzalny spadek mediany czasu o około 14,6% dla seed 42.
7. Static channel control osiągnął 55,19%, tworząc silny baseline dla dalszych
   eksperymentów strukturalnych.
8. Markery M1–M6 i audyt danych ujawniły ryzyka niewidoczne w globalnej
   accuracy.

---

## 11. Synteza wyników negatywnych

### 11.1. Wyniki negatywne potwierdzone eksperymentalnie

1. **Static/SE/Gumbel z `attention_v2` nie poprawiły CIFAR-10 accuracy.**
   Wszystkie sparowane delty były ujemne w seedzie 42.
2. **Proxy sparsity nie dało speedupu.** Gated modele były nieco wolniejsze od
   dense baseline, ponieważ konwolucje nadal były wykonywane.
3. **Threshold static L1 był niestabilny.** M3 fragile ratio wynosił około
   99,9–100% dla β>0.
4. **UCLA nie pokonał utility-only.** Pełny wynik to −1,76 pp z CI całkowicie
   poniżej zera.
5. **Uncertainty nie generalizowała jako sygnał ryzyka.** Korelacja z błędem
   wyniosła 0,054.
6. **Value-of-compute nie utrzymał dodatniego pilota.** Po 100 epokach średnia
   była o 0,28 pp gorsza od utility-only.
7. **Gęstsze targety nie naprawiły policy.** Agreement spadł do 7,19%.
8. **Oracle action cloning nie generalizowało.** Żaden z routerów nie przebił
   majority baseline na held-out danych.
9. **Adaptive depth nie ma jeszcze potwierdzonej przewagi nad static depth.**
   Wynik jest dodatni w seedzie 42 i ujemny w seedzie 123.
10. **Training-only semantic feedback pogorszył deployment o 5,36 pp.**

### 11.2. Czego nie wykonano albo nie domknięto

- wieloseedowej replikacji `attention_v2`;
- fizycznego channel pruning dla statycznych masek;
- deduplikowanej analizy wrażliwości CIFAR-100;
- pełnego dynamicznego BGA z `attention_v2` w jednolitym protokole;
- prawdziwego PTQ/QAT;
- ImageNet/ImageNet-100 dla finalnej metody;
- static-depth seed 2026;
- dłuższego 100/200-epokowego adaptive-depth;
- end-to-end dynamic batching i tail latency;
- pełnej analizy risk–coverage, ECE i spójności progów między wyjściami;
- fizycznie skompilowanego static channel modelu.

Brak tych elementów nie jest wynikiem pozytywnym ani negatywnym. Są to otwarte
pozycje, których nie należy przedstawiać jako „prawie potwierdzonych”.

---

## 12. Rola attention w projekcie

Attention występowało w kilku rolach:

1. w `attention_v2` jako uczona ważność kanałów SCA, SE, Gumbel i feedback;
2. w UCLA jako per-image scoring utility/uncertainty dla grup kanałów;
3. w value-of-compute jako policy przewidująca relatywną wartość etapów;
4. w semantic feedback jako głęboka korekta maski stage 1 podczas treningu.

Early exit sam w sobie nie jest channel attention. Jest mechanizmem
conditional computation. Można zbudować attention-based stopping controller,
ale samo dodanie modułu attention nie stanowi nowości i mogłoby ponownie
wprowadzić problem słabej generalizacji decyzji. Najpierw musi istnieć silny
baseline i jasno zdefiniowana informacja, której prosty confidence threshold
nie wykorzystuje.

W obecnym stanie najuczciwiej powiedzieć: główna linia channel attention dała
dobrze kontrolowane wyniki negatywne oraz narzędzia diagnostyczne; pozytywny
sygnał efektywności pochodzi z adaptive depth.

---

## 13. Ocena potencjału publikacyjnego

### 13.1. Sama nowa warstwa attention — niski potencjał

Static channel attention, SE, Gumbel gating, dynamic channel pruning i early
exit są dojrzałymi obszarami. BilevelPruning łączy static i dynamic channel
pruning, CNN Mixture-of-Depths bada selektywne przetwarzanie kanałów, a
[Pick-or-Mix](https://openaccess.thecvf.com/content/CVPR2024/html/Kumar_Pick-or-Mix_Dynamic_Channel_Sampling_for_ConvNets_CVPR_2024_paper.html)
proponuje dynamiczne próbkowanie kanałów. Claim „nowy attention do pruning”
byłby zbyt ogólny i najprawdopodobniej niewystarczający.

### 13.2. Framework M1–M6 — umiarkowany potencjał metodologiczny

Framework wiarygodności ma wartość edukacyjną i metodologiczną. Aby stał się
samodzielnym wkładem, wymaga:

- formalnego uzasadnienia każdego markera;
- badań na więcej niż CIFAR;
- niezbalansowanych danych i domain shift;
- porównania z istniejącymi metrykami calibration/fairness/robustness;
- wieloseedowych CI;
- pokazania, że markery zmieniają decyzję wyboru modelu albo przewidują
  problemy wdrożeniowe.

W obecnej postaci jest dobrym rozdziałem metodologicznym lub częścią większej
pracy, ale jeszcze nie zweryfikowaną nową metodą.

### 13.3. Wynik negatywny channel routing — umiarkowany potencjał

Kampania jest bardziej wartościowa niż pojedynczy nieudany eksperyment,
ponieważ zawiera pełny 200-epokowy test, matched utility-only, counterfactual
targets, duży oracle gap i held-out predictability w trzech punktach sieci.
Możliwa narracja brzmi:

> Label-informed channel allocation has large counterfactual potential, but
> its action is not predictable from inference-time features and standard
> uncertainty/value targets.

Do samodzielnej publikacji wynik negatywny wymagałby jednak mocniejszych
baseline'ów literaturowych, większego zbioru, wielu architektur i bardzo
precyzyjnej analizy, dlaczego oracle gap nie jest obserwowalny. Obecnie lepiej
wykorzystać go jako uzasadnienie pivotu i rozbudowany appendix.

### 13.4. Adaptive depth — najwyższy bieżący potencjał, ale nowość nieustalona

Adaptive depth ma najlepszy dowód praktyczny: trzy seedy, held-out kalibrację i
rzeczywisty prefix latency. Brakuje jednak przewagi nad static-depth oraz
unikalnego mechanizmu. Potencjalny wkład powinien koncentrować się na jednym z
trudniejszych problemów:

- kontrola ryzyka przy zadanym latency budget;
- spójna uncertainty między kolejnymi wyjściami;
- odporność progów na domain shift;
- tail latency P95/P99 w dynamicznych batchach;
- survivor-aware batching, gdzie po każdym exit batch jest kompaktowany;
- gwarancja jakości lub coverage przy limicie czasu.

Najbardziej interesująca z perspektywy dotychczasowych kompetencji projektu
jest **reliability- and tail-latency-aware adaptive inference**: połączenie
markerów wiarygodności z rzeczywistym wykonaniem early exit. To nadal wymaga
ukierunkowanego przeglądu literatury i nie może być deklarowane jako novum bez
porównania z aktualnymi pracami.

### 13.5. Static structural pruning — alternatywa o wysokiej wartości praktycznej

Static control 55,19% oraz wynik `attention_v2` β=0,5 sugerują, że stała
redundancja istnieje. Następny test powinien jednak fizycznie usuwać kanały.
Najpierw należy sprawdzić fixed-mask oracle: czy istnieje stały podzbiór 10/16
grup wyraźnie lepszy od uczonej maski. Jeżeli nie, dalsze skomplikowane
attention nie ma uzasadnienia. Jeżeli tak, można badać representation
preservation albo distillation dla statycznego, skompilowanego studenta.

Ten kierunek mocno nakłada się z BilevelPruning i klasycznym structured
pruning. Musi mieć bardzo konkretny wyróżnik i realne pomiary sprzętowe.

---

## 14. Rekomendowany dalszy plan

### Etap A — domknięcie obecnego dowodu adaptive depth

1. Uruchomić static-depth stage 3 dla seed 2026 z identycznym protokołem.
2. Policzyć średnią ± SD adaptive vs static dla trzech seedów.
3. Przygotować pełne krzywe accuracy–cost, a nie jeden próg 85%.
4. Raportować exit rates, ECE, NLL, Brier i risk–coverage per exit.
5. Zmierzyć end-to-end latency polityki, nie tylko prefixy.
6. Zmierzyć batch 1, 8, 32/64, medianę, P95 i P99 na GPU oraz CPU.
7. Sprawdzić batch compaction i koszt rozgałęzienia.

**Bramka go/no-go:** adaptive depth powinien tworzyć powtarzalny punkt Pareto
wobec full-depth i najlepszego static-depth, nie tylko wobec pełnej sieci.

### Etap B — dłuższy protokół

Jeżeli etap A przejdzie:

1. zamrozić 100-epokowy Tiny ImageNet protocol;
2. wykonać co najmniej trzy seedy;
3. dobrać progi wyłącznie na calibration split;
4. test wykorzystać jednokrotnie do finalnego audytu;
5. uwzględnić temperature scaling i silniejszy klasyfikator wyjść;
6. dopiero potem rozważyć ImageNet-100 lub ImageNet-1k.

20 epok pozostaje screeningiem. Wyniki z value-of-compute pokazały, że
przewaga pilota może zniknąć po 100 epokach. Dłuższy trening jest konieczny
przed publikacyjnym claimem.

### Etap C — decyzja o roli attention

Nie rekomenduję kolejnego per-image channel routera. Dopuszczalne są dwa
precyzyjne testy:

1. **Fixed-mask oracle** dla 10/16 grup stage 1. Jeśli najlepsza stała maska
   nie daje istotnego zysku nad 55,19%, zamknąć static attention/pruning.
2. **Attention stopping controller** tylko wtedy, gdy wykorzystuje informację
   nieobecną w confidence baseline i przechodzi held-out predictability audit
   przed pełnym treningiem.

Nie należy ponawiać training-only feedback bez mechanizmu eliminującego
train–deployment mismatch.

### Etap D — walidacja publikacyjna

- pełny przegląd literatury i tabela closest methods;
- co najmniej trzy seedy, najlepiej pięć dla finalnej tabeli;
- bootstrap lub przedziały dla różnic sparowanych;
- wspólny budżet treningowy i inferencyjny;
- full, static-depth, confidence, entropy i calibration baselines;
- rzeczywisty kod wykonania warunkowego;
- pomiary na jawnie określonym sprzęcie;
- ablation każdego nowego elementu;
- test domain shift lub co najmniej corruption benchmark;
- publiczny protokół reprodukcji.

---

## 15. Kryteria zatrzymania, aby unikać nieproduktywnego strojenia

### Zamknąć adaptive depth, jeżeli:

- po trzech seedach nie poprawia frontu Pareto względem static-depth;
- zysk proxy kosztu nie przekłada się na end-to-end latency;
- P95/P99 rośnie przez routing lub batch fragmentation;
- progi nie generalizują między splitami lub przy domain shift.

### Zamknąć static channel pruning, jeżeli:

- fixed-mask oracle nie daje wyraźnego potencjału;
- fizyczna kompilacja nie daje latency gain;
- wynik jest silnie zależny od progu lub seeda;
- standardowe structured-pruning baseline'y są równie dobre lub lepsze.

### Nie promować nowego attention, jeżeli:

- jego przewaga istnieje tylko na train lub best checkpoint;
- używa informacji niedostępnej w deployment;
- nie przebija prostego confidence/static baseline;
- dodaje koszt większy niż oszczędność;
- claim pokrywa się z istniejącą literaturą.

---

## 16. Ograniczenia całej kampanii

1. Znaczna część badań używa CIFAR-10, CIFAR-100 lub Tiny ImageNet, a nie
   pełnego ImageNet.
2. Część rezultatów `attention_v2` ma tylko jeden seed.
3. Lokalny CIFAR-100 zawiera 10 duplikatów train–test.
4. Wiele bramek jest tylko maskingiem, nie strukturalnym pruningiem.
5. Oracle korzysta z etykiety i jest wyłącznie górną granicą.
6. 512-elementowe audyty predictability są diagnostyczne, nie finalne.
7. Early exit był trenowany 20 epok, więc może zmienić relacje po zbieżności.
8. Static-depth nie ma jeszcze trzeciego seeda.
9. Latency early exit dotyczy dotąd batch size 1 i jednego checkpointu.
10. Nie przeprowadzono pełnej oceny domain shift, robustness ani fairness.
11. Nie wykonano prawdziwego PTQ/QAT.
12. Nowość publikacyjna nie została potwierdzona pełnym systematic review ani
    wyszukiwaniem patentowym.

---

## 17. Najważniejsze artefakty

### `attention_v2`

- `raport_badania.md` — audytowalny wynik CIFAR-10;
- `raport_dla_prowadzacego_2026-07-21_PL.md` — wiarygodność danych;
- `raport_markerow.md` — definicje M1–M6;
- `results_code_review/reproducible_audit_full.json` — pełny audyt modeli;
- `results_code_review/markers_full/reliability_report.json` — markery;
- `results_data_audit/data_credibility_audit.json` — integralność danych;
- `results_cifar100_seed42_4gpu/*.json` — pełne runy CIFAR-100.

### `attentionv3`

- `EXPERIMENT_RESULTS.md` — trzyseedowe piloty i global UCLA;
- `FAILURE_REPORT_GLOBAL_UCLA_SEED42.md` — analiza pełnego no-go;
- `VALUE_BUDGET_RESEARCH_LOG.md` — value-of-compute;
- `ORACLE_AUDIT_REPORT.md` — potencjał lokalnych transferów;
- `ORACLE_ACTION_PREDICTABILITY_REPORT.md` — globalna predykcja akcji;
- `FUTURE_STAGE_ROUTING_REPORT.md` — routing sekwencyjny;
- `EARLY_EXIT_REPLICATION_REPORT_2026-08-21.md` — trzy seedy early exit;
- `EARLY_EXIT_LATENCY_REPORT_2026-08-21.md` — prefix latency;
- `STATIC_DEPTH_SEED123_UPDATE_2026-08-21.md` — kontrola static-depth;
- `STATIC_CHANNEL_CONTROL_REPORT_2026-08-24.md` — static mask;
- `SEMANTIC_FEEDBACK_NEGATIVE_RESULT_2026-08-24.md` — feedback no-go.

Najważniejsze commity końcowej fazy:

- `426e576` — wynik global attention seed 42;
- `0199d55` — 100-epokowy protokół Tiny ImageNet;
- `31ee042` — oracle potential;
- `cfcb3d4` — zamknięcie channel routing;
- `f8e2d51` — adaptive-depth pilot;
- `19bd6ce` — replikacja early exit;
- `abe6cea` — realny prefix latency;
- `4af9c1f` — static-depth baseline;
- `a91cd23` — static channel control;
- `6955c3f` — negatywny wynik semantic feedback.

---

## 18. Pytania do omówienia z promotorem

1. Czy głównym celem publikacji ma być nowa metoda, czy rygorystyczna analiza
   granic dynamicznego channel routing?
2. Czy priorytetem ma być adaptive depth i real latency, czy fizyczny static
   structured pruning?
3. Czy framework M1–M6 ma być osobnym wkładem metodologicznym, czy narzędziem
   ewaluacji głównej metody?
4. Jaki zbiór ma stanowić finalną walidację: Tiny ImageNet, ImageNet-100 czy
   pełny ImageNet?
5. Czy dysponujemy zasobami na trzy pełne 100/200-epokowe seedy oraz benchmark
   wielourządzeniowy?
6. Czy wynik negatywny UCLA/oracle predictability powinien być częścią głównej
   narracji, czy appendixem dokumentującym pivot?

Moja rekomendacja jest następująca: główny tor powinien dotyczyć wiarygodnego
adaptive inference z rzeczywistym latency, a channel routing powinien zostać
wykorzystany jako dobrze udokumentowana motywacja i wynik negatywny. Static
structured pruning można prowadzić jako oddzielny, krótki feasibility track,
zaczynając od fixed-mask oracle.

---

## 19. Konkluzja końcowa

Projekt wykonał więcej niż serię prób z attention. Zbudował ścieżkę od miękkich
bramek bez kontroli kosztu do rygorystycznej ewaluacji decyzji adaptacyjnych.
Najważniejsze hipotezy channel routing zostały przetestowane uczciwie i w
większości odrzucone:

- uncertainty nie poprawiła alokacji;
- kontrfaktyczna value regression nie utrzymała przewagi po zbieżności;
- decyzja dobrego oracle'a nie była przewidywalna z cech inferencyjnych;
- głęboki semantic feedback zaszkodził statycznej ścieżce deploymentowej.

Nie oznacza to, że badania były bezowocne. Wyniki wykluczyły kilka pozornie
atrakcyjnych dróg, ujawniły krytyczne różnice między sparsity i szybkością oraz
doprowadziły do kierunku, który faktycznie pomija obliczenia. Adaptive depth
ma obecnie najlepszy stosunek jakości dowodu do potencjału praktycznego, ale
wymaga domknięcia static-depth, dłuższego treningu i benchmarku end-to-end.

Na dzień 24 sierpnia 2026 r. nie ma jeszcze podstaw do deklaracji gotowej nowej
metody publikacyjnej. Jest natomiast solidna, audytowalna kampania badawcza,
wyraźny wynik negatywny dla dynamicznego channel routing, wartościowy framework
diagnostyczny oraz obiecujący kierunek adaptive inference. Najbliższa decyzja
powinna opierać się na wyniku static-depth seed 2026 i pełnym porównaniu
accuracy–risk–latency.

---

## 20. Jednoznaczna karta wyników: co wyszło, co nie wyszło i co jest otwarte

Poniższa tabela jest skróconą odpowiedzią decyzyjną dla promotora. „Wyszło”
oznacza, że dany rezultat został rzeczywiście zmierzony i jest wsparty
odpowiednią kontrolą. Nie oznacza automatycznie gotowej nowości publikacyjnej.

| Element | Status | Konkretny dowód | Decyzja |
|---|---|---|---|
| Framework markerów M1–M6 | **WYSZŁO metodologicznie** | wykrywa degradację per klasa, bias błędów, threshold fragility, representation drift i ryzyko kwantyzacji | zachować jako warstwę ewaluacji |
| Static L1 CIFAR-100 | **WYSZEDŁ obiecujący pilot** | 71,59% vs 71,83% baseline przy 45,5% proxy sparsity | wymaga fizycznego pruning i wielu seedów |
| Exact global budget i soft quota | **WYSZŁO technicznie** | dokładnie 40/64 grup; usunięty kolaps greedy allocation | zachować infrastrukturę |
| Oracle transferów kanałów | **WYSZŁO jako górna granica** | +10,94 do +11,91 pp | dowodzi potencjału akcji, nie gotowej policy |
| Overfit zamrożonego targetu | **WYSZŁO diagnostycznie** | agreement 40,6%→76,6% | głowa i gradient działają |
| Early-exit oracle | **WYSZŁO** | 61,90–63,62% przy koszcie ok. 75–76% | uzasadniło budowę policy |
| Confidence early exit, 3 seedy | **WYSZŁO jako pilot** | −0,07 pp vs full przy koszcie 84,92% | kontynuować po static controls |
| Realny prefix latency | **WYSZŁO na batch 1** | 5,65→ok. 4,82 ms, około −14,6% | rozszerzyć na end-to-end i batching |
| Static channel baseline | **WYSZŁO jako kontrola** | 55,19%, best 56,26% | podstawa fixed-mask oracle |
| Static/SE/Gumbel przewyższa dense CIFAR-10 | **NIE WYSZŁO** | wszystkie delty ujemne; od −0,88 do −2,20 pp | nie claimować przewagi |
| Soft gating daje realny speedup | **NIE WYSZŁO** | gated latency była gorsza od dense | wymagany structural execution |
| Stabilny threshold static L1 | **NIE WYSZŁO** | fragile ratio 99,9–100% | nie opierać claimu na jednym τ |
| UCLA uncertainty routing | **NIE WYSZŁO** | 69,82% vs 71,58%; delta −1,76 pp | kierunek zamknięty |
| Kalibracja uncertainty UCLA | **NIE WYSZŁA** | korelacja uncertainty–błąd 0,054 | target i decyzja były niedopasowane |
| Value-of-compute | **NIE WYSZŁO po zbieżności** | 58,55% vs 58,83% po 100 epokach | nie skalować |
| Gęstszy kontrfaktyczny nadzór | **NIE WYSZEDŁ** | agreement 7,19% | sama częstotliwość nie wystarcza |
| Oracle action cloning | **NIE WYSZŁO** | stem/layer1/layer2 poniżej majority | channel routing zamknięty |
| Semantic-feedback static mask | **NIE WYSZŁO** | 49,83% vs 55,19%; −5,36 pp | nie replikować obecnej wersji |
| Adaptive depth > static depth | **NIEROZSTRZYGNIĘTE** | wygrana seed 42, przegrana seed 123 | wykonać seed 2026 |
| Publikacyjna nowość early exit | **NIEROZSTRZYGNIĘTE** | mechanizm bazowy jest znany | potrzebny konkretny claim risk/latency |
| Structural channel speedup | **NIEROZSTRZYGNIĘTE** | dotąd stosowano maskowanie | wykonać fixed-mask oracle i kompilację |

### Dalsze działania w kolejności wykonania

1. **Natychmiast:** ukończyć static-depth seed 2026 i policzyć trzyseedowe
   porównanie adaptive vs static.
2. **Następnie:** wykonać accuracy–cost curve, ECE/NLL/Brier, risk–coverage
   oraz end-to-end P50/P95/P99 dla batch 1/8/32 lub 64.
3. **Brama decyzyjna:** jeżeli adaptive depth nie tworzy powtarzalnego punktu
   Pareto ponad static-depth, zamknąć ten wariant przed długim treningiem.
4. **Jeżeli przejdzie:** zamrozić 100-epokowy, trzyseedowy protokół Tiny
   ImageNet; testować finalnie tylko po wyborze na validation/calibration.
5. **Osobny krótki tor attention:** fixed-mask oracle dla 10/16 grup. Tylko
   wyraźny oracle gap uzasadnia fizyczny structured pruning.
6. **Przed skalowaniem:** przygotować tabelę najbliższych metod i ustalić claim
   dotyczący reliability/tail latency, a nie samego „nowego attention”.
7. **Dopiero na końcu:** ImageNet-100/ImageNet, wiele urządzeń i pełny benchmark
   publikacyjny.
