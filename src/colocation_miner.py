import pandas as pd
from collections import defaultdict
from itertools import combinations
from scipy.spatial import KDTree
import time
from typing import Dict, List, Set, Tuple, Optional, Any

from src.colocation_pattern import ColocationPattern
from src.types import InstanceId, FeatureType, TypeInstancePair, Pattern, PatternInstance


class ColocationMiner:
    def __init__(self, radius: float = 0.005, min_prevalence: float = 0.3):
        """
        Initialize the colocation pattern miner.
        
        Args:
            radius: The neighborhood radius for spatial proximity
            min_prevalence: The minimum participation index threshold
        """
        self.radius = radius
        self.min_prevalence = min_prevalence
        self.patterns: List[ColocationPattern] = []
        
        self.spatial_indices: Dict[FeatureType, Dict[str, Any]] = {}
        self.instance_neighbors: Dict[TypeInstancePair, Set[TypeInstancePair]] = defaultdict(set)
        self.participation_ratios: Dict[Pattern, Dict[FeatureType, float]] = {}
        self.unique_types: List[FeatureType] = []
        self.instances_by_type: Dict[FeatureType, pd.DataFrame] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """
        Main method to find colocation patterns in spatial data.
        
        Args:
            df: DataFrame containing spatial features with columns: 'type', 'x', 'y'
        """
        start_time = time.time()
        
        self.df = df.reset_index(drop=True)
        self.df['id'] = self.df.index
        self.unique_types: List[FeatureType] = sorted(list(self.df["type"].unique()))
        self.instances_by_type: Dict[FeatureType, pd.DataFrame] = {
            t: self.df[self.df['type'] == t] for t in self.unique_types
        }
        print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        self._build_spatial_indices()
        print(f"Spatial indices built in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        self._precompute_all_neighbors()
        print(f"Neighbor precomputation completed in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        size_2_patterns = self._discover_size_2_patterns()
        self.patterns.extend(size_2_patterns)
        print(f"Found {len(size_2_patterns)} patterns of size 2 in {time.time() - start_time:.2f} seconds")
        
        k = 3
        while True:
            start_time = time.time()
            print(f"Processing patterns of length: {k}")
            
            candidates = self._generate_candidates(k)
            print(f"Found {len(candidates)} candidates")
            if not candidates:
                break
            
            new_patterns = self._discover_frequent_patterns_for_candidates(candidates)
            print(f"Found {len(new_patterns)} frequent patterns in {time.time() - start_time:.2f} seconds")
            
            if not new_patterns:
                break
                
            self.patterns.extend(new_patterns)
            k += 1

    def _build_spatial_indices(self) -> None:
        """Build KDTree indices for each type once to avoid rebuilding."""
        for t, instances in self.instances_by_type.items():
            points = instances[['x', 'y']].values
            self.spatial_indices[t] = {
                'tree': KDTree(points),
                'ids': instances['id'].values,
                'points': points
            }

    def _precompute_all_neighbors(self) -> None:
        """
        Precompute all neighbor relationships to avoid repeated spatial queries.
        For each instance, store all its neighbors regardless of type.
        """
        for t1, t2 in combinations(self.unique_types, 2):
            idx1 = self.spatial_indices[t1]
            idx2 = self.spatial_indices[t2]
            
            # find neighbors of all points of type t1 using the KDTree of type t2
            for i, point in enumerate(idx1['points']):
                id1 = idx1['ids'][i]
                neighbors = idx2['tree'].query_ball_point(point, self.radius)
                
                for j in neighbors:
                    id2 = idx2['ids'][j]

                    self.instance_neighbors[(t1, id1)].add((t2, id2))
                    self.instance_neighbors[(t2, id2)].add((t1, id1))

    def _discover_size_2_patterns(self) -> List[ColocationPattern]:
        """
        Discovers patterns of size 2 (pairs of types) with prevalence above threshold.
        Uses the precomputed neighbor relationships.
        
        Returns:
            List of ColocationPattern objects of size 2
        """
        patterns: List[ColocationPattern] = []
        
        for t1, t2 in combinations(self.unique_types, 2):
            participants_t1: Set[InstanceId] = set()
            participants_t2: Set[InstanceId] = set()
            
            instances: List[Tuple[InstanceId, InstanceId]] = []
            
            for id1 in self.instances_by_type[t1]['id'].values:
                has_neighbor = False
                for t, id2 in self.instance_neighbors.get((t1, id1), set()):
                    if t == t2:
                        has_neighbor = True
                        instances.append((id1, id2))
                        participants_t2.add(id2)
                
                if has_neighbor:
                    participants_t1.add(id1)
            
            pi1 = len(participants_t1) / len(self.instances_by_type[t1]) if len(self.instances_by_type[t1]) > 0 else 0
            pi2 = len(participants_t2) / len(self.instances_by_type[t2]) if len(self.instances_by_type[t2]) > 0 else 0
            pi = min(pi1, pi2)
            
            pattern = tuple(sorted([t1, t2]))
            self.participation_ratios[pattern] = {
                t1: pi1,
                t2: pi2
            }
            
            if pi >= self.min_prevalence and instances:
                patterns.append(ColocationPattern(pattern, pi, instances))
        
        return patterns

    def _generate_candidates(self, k: int) -> List[Pattern]:
        """
        Generate candidate patterns of size k using the apriori principle:
        All subsets of a frequent pattern must also be frequent.
        
        Args:
            k: Size of candidate patterns to generate
            
        Returns:
            List of candidate patterns of size k
        """
        
        prev_patterns = [p for p in self.patterns if len(p.types) == k-1]
        if len(prev_patterns) < 2:
            return []
        
        candidates: Set[Pattern] = set()
        
        for i, p1 in enumerate(prev_patterns):
            for p2 in prev_patterns[i+1:]:
                # join if k-2 items are the same
                if p1.types[:-1] == p2.types[:-1]:
                    new_candidate = tuple(sorted(set(p1.types) | set(p2.types)))
                    
                    valid = True
                    for subset in combinations(new_candidate, k-1):
                        subset = tuple(sorted(subset))
                        if not any(p.types == subset for p in prev_patterns):
                            valid = False
                            break
                    
                    if valid and len(new_candidate) == k:
                        candidates.add(new_candidate)
        
        return list(candidates)

    def _discover_frequent_patterns_for_candidates(self, candidates: List[Pattern]) -> List[ColocationPattern]:
        """
        Check which candidate patterns meet the minimum prevalence threshold.
        
        Args:
            candidates: List of candidate patterns to evaluate
            
        Returns:
            List of ColocationPattern objects that meet the prevalence threshold
        """
        new_patterns: List[ColocationPattern] = []
        
        for candidate in candidates:
            participants: Dict[FeatureType, Set[InstanceId]] = {t: set() for t in candidate}
            instances = self._find_pattern_instances(candidate)
            
            for instance in instances:
                for t, id_val in zip(candidate, instance):
                    participants[t].add(id_val)
            
            participation_ratios: Dict[FeatureType, float] = {}
            for t in candidate:
                participation_ratios[t] = (
                    len(participants[t]) / len(self.instances_by_type[t]) 
                    if len(self.instances_by_type[t]) > 0 else 0
                )
            
            self.participation_ratios[candidate] = participation_ratios

            pi = min(participation_ratios.values()) if participation_ratios else 0
            if pi >= self.min_prevalence and instances:
                new_patterns.append(ColocationPattern(candidate, pi, instances))
        
        return new_patterns

    def _find_pattern_instances(self, pattern_types: Tuple[Pattern]) -> List[PatternInstance]:
        """
        Find all instances of a given pattern type using precomputed neighbor information.
        
        Args:
            pattern_types: Tuple of feature types forming the pattern
            
        Returns:
            List of tuples containing instance IDs that form instances of the pattern
        """
        
        first_type = pattern_types[0]
        current_instances: List[Tuple[TypeInstancePair, ...]] = [
            ((first_type, id_val),) for id_val in self.instances_by_type[first_type]['id'].values
        ]
        
        for i in range(1, len(pattern_types)):
            current_type = pattern_types[i]
            new_instances: List[Tuple[TypeInstancePair, ...]] = []
            
            for instance in current_instances:
                can_extend = True
                for type_id_pair in instance:
                    # check if current instance has any neighbors of the current type
                    if not any(neighbor_type == current_type for neighbor_type, _ in 
                               self.instance_neighbors.get(type_id_pair, set())):
                        can_extend = False
                        break
                
                if not can_extend:
                    continue
                
                candidates: Set[InstanceId] | None = None
                
                for type_id_pair in instance:
                    # get neighbors of the current type for the current instance
                    current_neighbors = {neighbor_id for neighbor_type, neighbor_id in 
                                         self.instance_neighbors.get(type_id_pair, set()) 
                                         if neighbor_type == current_type}
                    
                    if candidates is None:
                        candidates = current_neighbors
                    else:
                        candidates &= current_neighbors
                    
                    if not candidates:
                        break
                
                if candidates:
                    for candidate_id in candidates:
                        new_instance = instance + ((current_type, candidate_id),)
                        new_instances.append(new_instance)
            
            if not new_instances:
                return []
            
            current_instances = new_instances
        
        result: List[PatternInstance] = []
        for instance in current_instances:
            id_list = [id_val for _, id_val in instance]
            result.append(tuple(id_list))
        
        return result

    def get_patterns(self) -> List[ColocationPattern]:
        """
        Returns all discovered patterns sorted by participation index.
        
        Returns:
            List of ColocationPattern objects sorted by descending PI and then by pattern size
        """
        return sorted(self.patterns, key=lambda p: (-p.pi, len(p.types)))
