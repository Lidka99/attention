# Future-stage routing audit — 2026-08-20

## Poprawny test po layer1

Po wykonaniu `layer1` zamrożono jego budżet. Oracle i student wybierali tylko
`noop` albo jeden z sześciu transferów między przyszłymi etapami 2, 3 i 4.
Student widział pooled feature po `layer1`. Protokół: 512 przykładów train,
512 validation, 300 kroków MLP.

| Split | Accuracy akcji | Majority baseline | Mean regret |
|---|---:|---:|---:|
| Train | 68,16% | 63,48% | 0,011 NLL |
| Validation | 31,64% | 33,98% | 1,410 NLL |

## Decyzja

Router po `layer1` nie generalizuje lepiej niż wybór klasy większościowej i
traci większość zysku oracle'a. Nie integrujemy go z ResNet.

Ostatnia wąska bramka w rodzinie sekwencyjnej to decyzja po `layer2`, z akcjami
tylko między etapami 3 i 4. Jej możliwy zysk jest mniejszy, ale decyzja ma
dostęp do bogatszych cech. Jeśli także nie przebije majority baseline'u,
zamyka to globalne i etapowe budżetowanie kanałów jako główny kierunek.
