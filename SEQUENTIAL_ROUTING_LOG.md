# Sequential routing log — 2026-08-19

## Niedopasowany audit layer1

Audyt z cech po `layer1` dla globalnej przestrzeni 13 akcji osiągnął 22,85%
accuracy na validation, poniżej majority baseline 23,83% (mean regret 1,559
NLL). Wynik nie jest rozstrzygający dla sekwencyjnego routingu.

Powód: globalny oracle może transferować budżet do lub z etapu 1, który po
obserwacji `layer1` jest już wykonany. Student sekwencyjny nie może wykonać
takiej akcji, więc jego feature i target nie opisują tego samego problemu.

## Właściwa następna bramka

Po `layer1` action space zawiera tylko `noop` oraz sześć kierunkowych
transferów między etapami 2, 3 i 4. Budżet etapu 1 zostaje zamrożony. Oracle
wybiera minimum NLL wyłącznie w tej przestrzeni, a policy widzi feature po
`layer1`. Dopiero ten held-out audit rozstrzyga, czy budować sekwencyjny
router dla przyszłych etapów.
