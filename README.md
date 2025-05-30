# Spatial Colocation Pattern Mining

This implementation is based on the research paper:

> Yan Huang, Shashi Shekhar, and Hui Xiong. "Discovering colocation patterns from spatial data sets: a general approach."

## Overview

Spatial colocation pattern mining identifies sets of features frequently located in close geographic proximity.

## Key Components

- **ColocationPattern**: Class representing a discovered pattern
- **ColocationMiner**: The main algorithm implementation
- **ColocationDataset**: Base data loader class for spatial data
- **OSMColocationDataset**: Data loader for OpenStreetMap data

## Algorithm Details

The implementation follows the paper's approach with these key steps:

1. **Neighbor Relationship Identification**: Find all pairs of instances that are within a specified distance threshold.
2. **Size-2 Pattern Discovery**: Find pairs of feature types with participation index above threshold.
3. **Candidate Generation**: Generate candidate patterns of size k using (k-1)-sized frequent patterns.
4. **Prevalence-Based Pruning**: Filter candidates using an upper bound of the participation index.
5. **Instance Discovery**: Find all instances of each candidate pattern.
6. **Prevalence Calculation**: Calculate the participation index of each pattern.

## Optimizations

This implementation includes several optimizations:

1. **Spatial Index Reuse**: KDTree indices are built once per feature type.
2. **Neighbor Precomputation**: All neighborhood relationships are computed and stored upfront.
3. **Clique-Based Instance Finding**: Uses clique-based approach to find pattern instances.

## Usage

### Basic Usage

```python
from src.colocation_dataset import OSMColocationDataset
from src.optimized_colocation_miner import ColocationMiner

# Define area and POI types
area = (52.229, 20.944, 52.410, 21.222)  # Approximate Warsaw area
poi_types = ["bar", "cafe", "restaurant", "pub"]

# Load data
dataset = OSMColocationDataset(area, poi_types)
data = dataset.load_data()

# Create miner and discover patterns
miner = ColocationMiner(radius=0.005, min_prevalence=0.5)
miner.fit(data)
patterns = miner.get_patterns()

# Display results
for pattern in patterns:
    print(pattern)
```

### CLI Testing

Use the included script to test the algorithm:

```bash
# Run
python main.py

# Change parameters
python main.py --optimized --radius 0.003 --min-prevalence 0.4 --poi-types bar,cafe,restaurant,pub
```

## Parameters

- **radius**: The neighborhood radius (distance threshold)
- **min_prevalence**: Minimum participation index threshold
