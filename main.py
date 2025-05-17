import pandas as pd
import time
import argparse

from src.colocation_dataset import OSMColocationDataset
from src.colocation_miner import ColocationMiner


def main():
    parser = argparse.ArgumentParser(description='Run colocation pattern mining on OSM data')
    parser.add_argument('--radius', type=float, default=0.005, help='Neighborhood radius')
    parser.add_argument('--min-prevalence', type=float, default=0.5, help='Minimum participation index')
    parser.add_argument('--area', type=str, default='52.229,20.944,52.410,21.222', 
                        help='Bounding box in format "min_lat,min_lon,max_lat,max_lon"')
    parser.add_argument('--poi-types', type=str, 
                        default='bar,cafe,fast_food,food_court,pub,restaurant',
                        help='Comma-separated list of POI types')
    args = parser.parse_args()

    # Parse area
    area = tuple(map(float, args.area.split(',')))
    assert len(area) == 4, "Area must be specified as min_lat,min_lon,max_lat,max_lon"
    
    # Parse POI types
    # poi_types = args.poi_types.split(',')
    poi_types = poi_types = ["bar", "cafe", "fast_food", "food_court", "ice_cream", "pub", 'restaurant', "college", "library", "research_institute", "school", "university", "parking", "atm", 'bank', "clinic", "doctors", "pharmacy", "veterinary", "casino", "cinema", "events_venue", "nightclub", "theatre", "	police"]
    
    print(f"Loading data for {len(poi_types)} POI types in area {area}")
    dataset = OSMColocationDataset(area, poi_types)
    data = dataset.load_data()
    
    print(f"Loaded {len(data)} points")
    for poi_type in poi_types:
        count = len(data[data['type'] == poi_type])
        print(f"  - {poi_type}: {count} instances")
    
    start_time = time.time()
    miner = ColocationMiner(radius=args.radius, min_prevalence=args.min_prevalence)
    miner.fit(data)
    total_time = time.time() - start_time
    
    patterns = miner.get_patterns()
    patterns_df = pd.DataFrame([p.to_dict() for p in patterns])
    
    print(f"\nMining completed in {total_time:.2f} seconds")
    print(f"Found {len(patterns)} patterns")
    
    if len(patterns) > 0:
        print("\nTop patterns:")
        pd.set_option('display.max_colwidth', None)
        print(patterns_df.head(10))

if __name__ == "__main__":
    main()
