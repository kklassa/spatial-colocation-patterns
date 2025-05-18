from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd
import overpy


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
            DataFrame with the loaded data.
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
