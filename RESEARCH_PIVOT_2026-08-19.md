# Pivot badawczy — oracle-distilled budget routing

## Decyzja

Kończymy linię **stage value-of-compute regression**. Nie przechodzi ona
kryterium jakości: po 100 epokach value-of-compute przegrał z utility-only
średnio o 0,28 pp na seedach 42 i 123. Ani rzadszy, ani liczony co batch
kontrfaktyczny nadzór nie nauczył rankingu etapów; w pilocie gęstego targetu
agreement wyniósł tylko 7,19%.

Nie jest to jednak brak potencjału samej alokacji. Label-informed oracle
lokalnych transferów poprawił Top-1 o 10,94--11,91 pp na 512 przykładach
walidacyjnych. Luka jest więc między wyborem oracle'a a polityką, nie w
przestrzeni legalnych transferów.

## Nowa hipoteza

**Oracle-Distilled Budget Routing (ODBR):** zamiast przewidywać cztery wartości
etapów, kontroler klasyfikuje jawną akcję budżetową:

- `noop`;
- 12 kierunkowych transferów `donor -> recipient` po cztery grupy.

Podczas treningu nauczyciel/oracle wybiera akcję o najmniejszym NLL spośród
legalnych akcji. Student dostaje cross-entropy tej akcji oraz zwykłą stratę
klasyfikacji. Akcje nielegalne są maskowane przed softmaxem.

## Bramka przed integracją

Nie wdrażamy od razu kolejnego pełnego treningu. Najpierw wykonujemy tani,
zamrożony audyt przewidywalności:

1. nauczyciel generuje oracle-action label dla train i validation;
2. klasyfikator akcji widzi tylko cechy dostępne w chwili decyzji;
3. raportujemy accuracy względem większościowej klasy i mean regret względem
   oracle'a na held-out split.

Jeżeli predykcja nie przebije majority baseline'u albo regret nie maleje,
cechy po stemie są niewystarczające. Wtedy przechodzimy do decyzji sekwencyjnej
po kolejnych etapach ResNet, a nie strojenia kolejnego globalnego kontrolera.

## Co pozostaje niezmienione

- globalny budżet 40/64 i floor cztery grupy;
- utility-only jako najważniejsza kontrola;
- brak claimu o speedupie, dopóki kanały nie są wykonywane strukturalnie.
