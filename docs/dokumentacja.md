# Algorytm Odkrywania Wzorców Kolokacji

## Wprowadzenie

Algorytm do odkrywania wzorców kolokacji w danych przestrzennych został zaimplementowany na podstawie artykułu "Discovering colocation patterns from spatial data sets: a general approach" autorstwa Yan Huang, Shashi Shekhar i Hui Xiong. Głównym celem algorytmu jest identyfikacja zbiorów cech przestrzennych (typów obiektów), które często występują blisko siebie w przestrzeni geograficznej.

## Klasa `ColocationPattern`

Klasa `ColocataionPattern` reprezentuje jeden wzorzec kolokacji, zawierając cechy przestrzenne wchodzące w skład wzorca, wartość wskaźnika uczestnictwa (participation index) dla wzorca oraz listę instancji kolokacji mu odpowiadających. 

### Incjalizacja

```python
def __init__(
   self, 
   types: Tuple[FeatureType], 
   participation_index: float, 
   instances: List[PatternInstance],
):
```

**Parametry:**
- `types`: cechy przestrzenne wchodzące w skład wzorca.
- `participation_index`: wskaźnik uczestnictwa dla wzorca.
- `instances`: lista instacji kolokacji odpowiadających wzorcowi.

**Opis:**
Konstruktor inicjalizuje obiekt na podstawie danych odkrytego przez algorytm wzorca.

### Metoda `__str__`

```python
def __str__(self) -> str:
```

**Opis:**
Zwraca tekstową reprezentację wzorca.

### Metoda `to_dict`

```python
def to_dict(self) -> Dict[str, object]:
```

**Opis:**

Konwertuje wzorzec do reprezentacji słownikowej i zwraca ją. Słownik zawiera klucze `types`, `participation_index` oraz `num_instances`.

## Klasa `ColocationMiner`

Klasa `ColocationMiner` jest głównym komponentem implementacji, odpowiedzialnym za odkrywanie wzorców kolokacji z danych przestrzennych.

### Inicjalizacja

```python
def __init__(self, radius: float = 0.005, min_prevalence: float = 0.3):
```

**Parametry:**
- `radius`: Promień sąsiedztwa (odległość progowa) określający, kiedy dwa obiekty są uznawane za sąsiadujące.
- `min_prevalence`: Minimalny próg wskaźnika uczestnictwa (participation index) - miara częstości występowania wzorca.

**Opis:**
Konstruktor inicjalizuje podstawowe parametry oraz struktury danych wykorzystywane w algorytmie:
- `patterns`: Lista odkrytych wzorców kolokacji.
- `spatial_indices`: Słownik indeksów przestrzennych (KDTree) dla każdego typu obiektu.
- `instance_neighbors`: Słownik przechowujący relacje sąsiedztwa dla wszystkich instancji.
- `participation_ratios`: Słownik przechowujący współczynniki uczestnictwa dla każdego typu w każdym wzorcu.

### Metoda `fit`

```python
def fit(self, df: pd.DataFrame) -> None:
```

**Parametry:**
- `df`: DataFrame zawierający dane przestrzenne z kolumnami 'type', 'x', 'y'.

**Opis:**
Główna metoda uruchamiająca proces odkrywania wzorców kolokacji. Proces przebiega w następujących krokach:
1. Przygotowanie danych (dodanie identyfikatorów, grupowanie instancji według typów).
2. Budowa indeksów przestrzennych dla każdego typu obiektu.
3. Obliczenie wszystkich relacji sąsiedztwa.
4. Odkrycie wzorców rozmiaru 2 (pary typów).
5. Iteracyjne odkrywanie coraz większych wzorców (k > 2):
   - Generowanie kandydatów rozmiaru k.
   - Przycinanie kandydatów na podstawie górnych ograniczeń wskaźnika uczestnictwa.
   - Odkrywanie instancji pozostałych kandydatów.
   - Obliczanie wskaźników uczestnictwa i filtrowanie według progu.

### Metoda `_build_spatial_indices`

```python
def _build_spatial_indices(self) -> None:
```

**Opis:**
Tworzy indeksy przestrzenne (KDTree) dla każdego typu obiektu. Dla każdego typu obiektu:  
1. Pobiera współrzędne wszystkich instancji tego typu.
2. Buduje drzewo KDTree dla tych współrzędnych.
3. Przechowuje drzewo, identyfikatory instancji i współrzędne w słowniku `spatial_indices`.

Ta optymalizacja eliminuje potrzebę wielokrotnego budowania drzew KDTree podczas wykonywania algorytmu.

### Metoda `_precompute_all_neighbors`

```python
def _precompute_all_neighbors(self) -> None:
```

**Opis:**
Oblicza wszystkie relacje sąsiedztwa z góry (w ramach optymalizacji algorytmu). Dla każdej pary typów obiektów:
1. Używa wcześniej zbudowanych drzew KDTree do znalezienia wszystkich par instancji, które są w odległości mniejszej niż `radius`.
2. Zapisuje relacje sąsiedztwa w macierzy sąsiedztwa `instance_neighbors`.

### Metoda `_discover_size_2_patterns`

```python
def _discover_size_2_patterns(self) -> List[ColocationPattern]:
```

**Opis:**
Odkrywa wzorce kolokacji rozmiaru 2 (pary typów obiektów) o wskaźniku uczestnictwa powyżej progu. Dla każdej pary typów obiektów, metoda:
1. Identyfikuje instancje obu typów, które uczestniczą we wzorcu (mają sąsiada drugiego typu).
2. Oblicza współczynniki uczestnictwa dla obu typów (jaki procent instancji każdego typu ma sąsiada drugiego typu).
3. Oblicza wskaźnik uczestnictwa jako minimum z tych współczynników.
4. Jeśli wskaźnik jest powyżej progu, tworzy nowy wzorzec kolokacji.
5. Zapisuje współczynniki uczestnictwa dla późniejszego wykorzystania w przycinaniu na podstawie wskaźnika uczestnictwa.

### Metoda `_generate_candidates`

```python
def _generate_candidates(self, k: int) -> List[Pattern]:
```

**Parametry:**
- `k`: Rozmiar generowanych kandydatów.

**Opis:**
Generuje kandydatów na wzorce kolokacji rozmiaru k, wykorzystując zasadę apriori: wszystkie podzbiory częstego wzorca muszą być również częste. Metoda:
1. Pobiera wszystkie wzorce rozmiaru k-1 z poprzedniej iteracji.
2. Dla każdej pary wzorców, które mają identyczne k-2 pierwsze typy, łączy je, tworząc potencjalnego kandydata rozmiaru k.
3. Weryfikuje, czy wszystkie podzbiory rozmiaru k-1 potencjalnego kandydata są częstymi wzorcami.
4. Jeśli tak, dodaje kandydata do listy.

### Metoda `_discover_frequent_patterns_for_candidates`

```python
def _discover_frequent_patterns_for_candidates(self, candidates: List[Pattern]) -> List[ColocationPattern]:
```

**Parametry:**
- `candidates`: Lista kandydatów na wzorce do oceny.

**Opis:**
Sprawdza, którzy kandydaci spełniają kryterium minimalnego wskaźnika uczestnictwa. Dla każdego kandydata, metoda:
1. Znajduje wszystkie instancje wzorca za pomocą metody `_find_pattern_instances`.
2. Dla każdego typu w kandydacie, zlicza liczbę instancji, które uczestniczą we wzorcu.
3. Oblicza współczynniki uczestnictwa dla każdego typu.
4. Oblicza wskaźnik uczestnictwa jako minimum z tych współczynników.
5. Jeśli wskaźnik jest powyżej progu, tworzy nowy wzorzec kolokacji.
6. Zapisuje współczynniki uczestnictwa dla późniejszego wykorzystania w przycinaniu.

### Metoda `_find_pattern_instances`

```python
def _find_pattern_instances(self, pattern_types: Tuple[Pattern]) -> List[PatternInstance]:
```

**Parametry:**
- `pattern_types`: Krotka typów tworzących wzorzec.

**Opis:**
Znajduje wszystkie instancje danego wzorca za pomocą podejścia opartego na klikach w grafie sąsiedztwa. Metoda wykorzystuje wcześniej obliczone relacje sąsiedztwa. Jej działanie przebiega następująco:
1. Rozpoczyna od instancji pierwszego typu w wzorcu.
2. Iteracyjnie dodaje jeden typ na raz:
   - Dla każdej częściowej instancji, sprawdza, czy może zostać rozszerzona o instancje bieżącego typu.
   - Znajduje wszystkie instancje bieżącego typu, które są sąsiadami wszystkich elementów częściowej instancji.
   - Tworzy nowe częściowe instancje przez dodanie tych kandydatów.
3. Jeśli na dowolnym etapie nie można znaleźć rozszerzeń, wzorzec nie ma instancji.
4. Ekstrahuje same identyfikatory z końcowych instancji.

### Metoda `get_patterns`

```python
def get_patterns(self) -> List[ColocationPattern]:
```

**Opis:**
Zwraca wszystkie odkryte wzorce kolokacji posortowane według wskaźnika uczestnictwa (malejąco) i rozmiaru wzorca.

## Klasa `ColocationDataset`

Bazowa, abstrakcyjna klasa zbioru danych umożliwiającego wykorzystanie do odkrywania wzorców kolokacji przestrzennych.

### Inicjalizacja

```python
def __init__(self):
```

**Opis:**
Konstruktor inicjalizuje obiekt oraz domyślnie ustawia atrybut `_data` na `None`. 

### Metoda `load_data`

```python
@abstractmethod
def load_data(self) -> pd.DataFrame:
```

**Opis:**
Abstrakcyjna metoda wczytująca dane ze źródła danych i zwracająca `DataFrame` z wczytanymi danymi. 

## Klasa `OSMColocationDataset`

Klasa `OSMColocationDataset` jest implementacją zbioru danych do odkrywania kolokacj przestrzennych bazującego na danych o points of interest (POI) dostępnych w bazie danych geograficznych OpenStreetMap.

### Inicjalizacja

```python
def __init__(self, area: Tuple[float], poi_types: List[str]):
```

**Parametry:**
- `area`: Krotka zawierające współrzędne bounding box terenu w kolejności minimalna szerokość, minimalna długość, maksymalna szerokość, masymalna długość geograficzna.
- `poi_types`: Lista wybrancyh typów POI dostępnych w OpenStreetMap (możliwe typy można znaleźć [pod tym adresem](https://wiki.openstreetmap.org/wiki/Pl:Key:amenity)).

**Opis:**
Konstruktor inicjalizuje obiekt oraz atrybuty klasy.

### Metoda `load_data`

```python
def load_data(self) -> pd.DataFrame:
```

**Opis:**
Metoda wykorzystująca bilbiotekę `overpy` do odpytania bazy danych OSM o położenie POI występujących na obszarze podanym przy inicjalizacji klasy. Dla każdego POI zwracany jest identyfikator `id`, nazwa typu `type` oraz współrzędna szerokości geograficznej `x` oraz długości geograficznej `y`.

## Klasa `GBIFColocationDataset`

Klasa `GBIFColocationDataset` jest implementacją zbioru danych do odkrywania kolokacj przestrzennych bazującego na danych o obserwacjach gatunków roślin i zwierząt zanotowanych w bazie danych Global Biodiversity Information Facility (GBIF).

### Inicjalizacja

```python
def __init__(self, 
   area: Tuple[float], 
   species_names: List[str], 
   min_year: int = 2010,
   limit_per_species: int | None = None,
):
```

**Parametry:**
- `area`: Krotka zawierające współrzędne bounding box terenu w kolejności minimalna szerokość, minimalna długość, maksymalna szerokość, masymalna długość geograficzna.
- `species_names`: Lista nazw naukowych wybranych gatunków.
- `min_year`: Minimalny rok obserwacji do włączenia w zbiór danych.
- `limit_per_species`: Opcjonalny limit ilości obserwacji dla jednego gatunku w zbiorze.

**Opis:**
Konstruktor inicjalizuje obiekt oraz atrybuty klasy. 

### Metoda `load_data`

```python
def load_data(self) -> pd.DataFrame:
```

**Opis:**
Metoda wywołuje metody `_get_species_key` oraz `_get_all_occurrences` dla każdego z podanych gatunków i zapisuje pobrane obserwacje w DataFrame. Dla obserwacji gatunku zwracany jest identyfikator `id`, nazwa gatunku `type` oraz współrzędna szerokości geograficznej `x` oraz długości geograficznej `y` na której dokonano obserwacji.

### Metoda `_get_species_key`

```python
def _get_species_key(self, species_name: str) -> int:
```

**Parametry:**
- `species_name`: Nazwa naukowa gatunku rośliny lub zwierzęcia.

**Opis:**
Metoda wykorzystuje API GBIF do odnalezienia klucza w bazie danych odpowiadającego wybranemu gatunkowi.

### Metoda `_get_all_occurrences`

```python
def _get_all_occurrences(self, species_key: int, species_name: str) -> List[Dict[str, Any]]:
```

**Parametry:**
- `species_key`: klucz w bazie GBIF odpowiadający gatunkowi.
- `species_name`: Nazwa naukowa gatunku rośliny lub zwierzęcia.

**Opis:**
Metoda wykorzystuje API GBIF do pobrania wszystkich (lub ograniczonych limitem) obserwacji danego gatunku na obszarze ustawionym podczas inicjalizacji klasy. 
