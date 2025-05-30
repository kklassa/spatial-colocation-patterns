import requests
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple, Dict, Any

import overpy
import pandas as pd


class ColocationDataset(ABC):
    def __init__(self):
        """
        Base class for colocation datasets.
        """
        self._data = None

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Loads the data from the source.
        
        Returns:
            DataFrame with the loaded data.
        """
        pass

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the loaded data.
        
        Returns:
            DataFrame with the loaded data.
        """
        if self._data is None:
            self.load_data()
        return self._data


class OSMColocationDataset(ColocationDataset):
    def __init__(self, area: Tuple[float], poi_types: List[str]):
        """
        Colocation dataset for OpenStreetMap (OSM) data.

        Args:
            area (tuple): Bounding box in the format (min_lat, min_lon, max_lat, max_lon).
            poi_types (list): List of POI types to load from OSM.
        """
        super().__init__()
        self._area = area
        self._poi_types = poi_types

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from OSM using the Overpass API."
        
        Returns:
            DataFrame with the generated data in the format:
            - id: Instance ID
            - type: Feature type
            - x: X coordinate
            - y: Y coordinate
        """
        api = overpy.Overpass()

        query = f"""
        [out:json];
        (
            {' '.join([f'node["amenity"="{poi}"]({self._area[0]},{self._area[1]},{self._area[2]},{self._area[3]});' for poi in self._poi_types])}
        );
        out body;
        """

        result = api.query(query)

        data = []
        for node in result.nodes:
            data.append({
                "id": node.id,
                "type": node.tags.get('amenity', 'unknown'),
                "x": node.lat,
                "y": node.lon
            })

        self._data = pd.DataFrame(data)
        return self.data


class GBIFColocationDataset(ColocationDataset):
    def __init__(self, 
        area: Tuple[float], 
        species_names: List[str], 
        min_year: int = 2010,
        limit_per_species: int | None = None,
    ):
        """
        Colocation dataset for Global Biodiversity Information Facility (GBIF) data.

        Args:
            area (tuple): Bounding box in the format (min_lat, min_lon, max_lat, max_lon).
            species_names (list): List of species scientific names to load from GBIF.
            min_year (int): Minimum year for data if recent_years_only is True.
            limit_per_species (int): Maximum number of records per species.
        """
        super().__init__()
        self._area = area
        self._species_names = species_names
        self._min_year = min_year
        self._limit_per_species = limit_per_species

    def load_data(self) -> pd.DataFrame:
        """
        Load species occurrence data from GBIF API.
        
        Returns:
            DataFrame with the generated data in the format:
            - id: Instance ID
            - type: Feature type
            - x: X coordinate
            - y: Y coordinate
        """
        data = []
        
        for species_name in self._species_names:
            print(f"\nProcessing species: {species_name}")
            
            species_key = self._get_species_key(species_name)
            
            if species_key:
                print(f"Found species key: {species_key}")
                
                species_data = self._get_all_occurrences(species_key, species_name)
                print(f"Retrieved {len(species_data)} occurrences")
                
                data.extend(species_data)
                
                time.sleep(1)
            else:
                print(f"Warning: Could not find species '{species_name}' in GBIF database")
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            self._data = df[["id", "type", "x", "y"]]
        else:
            print("No data found for any species in the specified area")
            self._data = pd.DataFrame(columns=["id", "type", "x", "y"])
            
        return self._data
        
    def _get_species_key(self, species_name: str) -> int:
        """
        Get the GBIF taxon key for a species name.
        
        Args:
            species_name (str): Scientific name of the species.
        
        Returns:
            GBIF taxon key for the species, or None if not found.
        """
        url = "https://api.gbif.org/v1/species/match"
        params = {"name": species_name, "strict": "false"}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data.get('matchType') not in ['NONE', None]:
                return data.get('usageKey')
            print(f"Warning: No match found for {species_name}. Response: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Error querying species {species_name}: {e}")
        
        return None
    
    def _get_all_occurrences(self, species_key: int, species_name: str) -> List[Dict[str, Any]]:
        """
        Get all occurrence records for a species with pagination and filtering.
        
        Args:
            species_key (int): GBIF taxon key for the species.
            species_name (str): Scientific name of the species.

        Returns:
            List of occurrence records with keys: id, type, x, y, year, month, day.
        """
        min_lat, min_lon, max_lat, max_lon = self._area
        
        base_params = {
            "taxonKey": species_key,
            "hasCoordinate": "true",
            "decimalLatitude": f"{min_lat},{max_lat}",
            "decimalLongitude": f"{min_lon},{max_lon}",
            "limit": 300,  # Max per page
            "year": f"{self._min_year},{datetime.now().year}",
        }

        url = "https://api.gbif.org/v1/occurrence/search"
        all_occurrences = []
        offset = 0
        total_count = None
        
        try:
            params = {**base_params, "offset": 0}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            total_count = data.get('count', 0)
            
            print(f"Found {total_count} records for {species_name}")
            
            if self._limit_per_species is not None:
                records_to_fetch = min(total_count, self._limit_per_species)
            else:
                records_to_fetch = total_count
            
            while len(all_occurrences) < records_to_fetch and offset < total_count:
                params = {**base_params, "offset": offset}
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for result in data.get('results', []):
                    if result.get('decimalLatitude') is not None and result.get('decimalLongitude') is not None:
                        all_occurrences.append({
                            "id": str(result.get('key')),
                            "type": species_name,
                            "x": float(result.get('decimalLatitude')),
                            "y": float(result.get('decimalLongitude')),
                            "year": result.get('year'),
                            "month": result.get('month'),
                            "day": result.get('day')
                        })
                
                offset += 300
                if len(all_occurrences) >= records_to_fetch:
                    break
                
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error retrieving occurrences for {species_name}: {e}")
        
        return all_occurrences[:records_to_fetch]
