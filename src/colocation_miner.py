import pandas as pd
from collections import defaultdict
from itertools import combinations
from scipy.spatial import KDTree
import time

from src.colocation_pattern import ColocationPattern


class ColocationMiner:
    def __init__(self, radius=0.005, min_prevalence=0.3):
        self.radius = radius
        self.min_prevalence = min_prevalence
        self.patterns = []
        # Store spatial indices for reuse
        self.spatial_indices = {}
        # Store neighbor relations for all instances
        self.instance_neighbors = defaultdict(set)

    def fit(self, df: pd.DataFrame):
        """
        Main method to find colocation patterns in spatial data.
        """
        start_time = time.time()
        
        # Prepare data
        self.df = df.reset_index(drop=True)
        self.df['id'] = self.df.index
        self.unique_types = list(self.df['type'].unique())
        self.instances_by_type = {
            t: self.df[self.df['type'] == t] for t in self.unique_types
        }
        
        print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        
        # Build the spatial index once for each type
        self._build_spatial_indices()
        print(f"Spatial indices built in {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        
        # Find all neighbor pairs once
        self._precompute_all_neighbors()
        print(f"Neighbor precomputation completed in {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        
        # Find size-2 patterns
        size_2_patterns = self._discover_size_2_patterns()
        self.patterns.extend(size_2_patterns)
        print(f"Found {len(size_2_patterns)} patterns of size 2 in {time.time() - start_time:.2f} seconds")
        
        # Process larger patterns iteratively
        k = 3
        while True:
            start_time = time.time()
            print(f"Processing patterns of length: {k}")
            
            # Generate candidate patterns of size k
            candidates = self._generate_candidates(k)
            print(f"Found {len(candidates)} candidates")
            if not candidates:
                break
            
            # Find which candidates meet the prevalence threshold
            new_patterns = self._discover_frequent_patterns_for_candidates(candidates)
            print(f"Found {len(new_patterns)} frequent patterns in {time.time() - start_time:.2f} seconds")
            
            if not new_patterns:
                break
                
            self.patterns.extend(new_patterns)
            k += 1

    def _build_spatial_indices(self):
        """Build KDTree indices for each type once to avoid rebuilding"""
        for t, instances in self.instances_by_type.items():
            points = instances[['x', 'y']].values
            self.spatial_indices[t] = {
                'tree': KDTree(points),
                'ids': instances['id'].values,
                'points': points
            }

    def _precompute_all_neighbors(self):
        """
        Precompute all neighbor relationships to avoid repeated spatial queries.
        For each instance, store all its neighbors regardless of type.
        """
        # For each pair of types
        for t1, t2 in combinations(self.unique_types, 2):
            idx1 = self.spatial_indices[t1]
            idx2 = self.spatial_indices[t2]
            
            # Query all points of type t1 against the KDTree of type t2
            for i, point in enumerate(idx1['points']):
                id1 = idx1['ids'][i]
                neighbors = idx2['tree'].query_ball_point(point, self.radius)
                
                # For each neighbor found
                for j in neighbors:
                    id2 = idx2['ids'][j]
                    # Store bidirectional relationship
                    self.instance_neighbors[(t1, id1)].add((t2, id2))
                    self.instance_neighbors[(t2, id2)].add((t1, id1))

    def _discover_size_2_patterns(self):
        """
        Discovers patterns of size 2 (pairs of types) with prevalence above threshold.
        Uses the precomputed neighbor relationships.
        """
        patterns = []
        
        for t1, t2 in combinations(self.unique_types, 2):
            # Find participating instances of each type
            participants_t1 = set()
            participants_t2 = set()
            
            # Count instances that participate in this relationship
            instances = []
            
            for id1 in self.instances_by_type[t1]['id'].values:
                has_neighbor = False
                for t, id2 in self.instance_neighbors.get((t1, id1), set()):
                    if t == t2:
                        has_neighbor = True
                        instances.append((id1, id2))
                        participants_t2.add(id2)
                
                if has_neighbor:
                    participants_t1.add(id1)
            
            # Calculate participation indices
            pi1 = len(participants_t1) / len(self.instances_by_type[t1]) if len(self.instances_by_type[t1]) > 0 else 0
            pi2 = len(participants_t2) / len(self.instances_by_type[t2]) if len(self.instances_by_type[t2]) > 0 else 0
            pi = min(pi1, pi2)
            
            # If prevalence is above threshold, add to patterns
            if pi >= self.min_prevalence and instances:
                patterns.append(ColocationPattern((t1, t2), pi, instances))
        
        return patterns

    def _generate_candidates(self, k):
        """
        Generate candidate patterns of size k using the apriori principle:
        All subsets of a frequent pattern must also be frequent.
        """
        # Get patterns of size k-1 from previous iteration
        prev_patterns = [p for p in self.patterns if len(p.types) == k-1]
        if len(prev_patterns) < 2:
            return []
        
        # Use a set to avoid duplicates
        candidates = set()
        
        for i, p1 in enumerate(prev_patterns):
            for p2 in prev_patterns[i+1:]:
                # If k-2 items are the same, join to create a candidate of size k
                if p1.types[:-1] == p2.types[:-1]:
                    # Join and sort to ensure consistent ordering
                    new_candidate = tuple(sorted(set(p1.types) | set(p2.types)))
                    
                    # Apply apriori principle: check if all subsets are frequent
                    valid = True
                    for subset in combinations(new_candidate, k-1):
                        subset = tuple(sorted(subset))
                        if not any(p.types == subset for p in prev_patterns):
                            valid = False
                            break
                    
                    if valid and len(new_candidate) == k:
                        candidates.add(new_candidate)
        
        return list(candidates)

    def _discover_frequent_patterns_for_candidates(self, candidates):
        """
        Check which candidate patterns meet the minimum prevalence threshold.
        """
        new_patterns = []
        
        for candidate in candidates:
            # For each type in the candidate pattern, count participating instances
            participants = {t: set() for t in candidate}
            instances = self._find_pattern_instances(candidate)
            
            # Count participating instances for each type
            for instance in instances:
                for t, id_val in zip(candidate, instance):
                    participants[t].add(id_val)
            
            # Calculate participation index
            pis = [
                len(participants[t]) / len(self.instances_by_type[t]) 
                for t in candidate 
                if len(self.instances_by_type[t]) > 0
            ]
            pi = min(pis) if pis else 0
            
            # Add to patterns if above threshold
            if pi >= self.min_prevalence and instances:
                new_patterns.append(ColocationPattern(candidate, pi, instances))
        
        return new_patterns

    def _find_pattern_instances(self, pattern_types):
        """
        Find all instances of a given pattern type using precomputed neighbor information.
        This is a critical performance method using clique-based approach.
        """
        # Start with instances of the first type
        first_type = pattern_types[0]
        current_instances = [((first_type, id_val),) for id_val in self.instances_by_type[first_type]['id'].values]
        
        # Iteratively add one type at a time
        for i in range(1, len(pattern_types)):
            current_type = pattern_types[i]
            new_instances = []
            
            for instance in current_instances:
                # Check if this partial instance can be extended
                can_extend = True
                for type_id_pair in instance:
                    # Check if all existing members neighbor the current type
                    if not any(neighbor_type == current_type for neighbor_type, _ in 
                               self.instance_neighbors.get(type_id_pair, set())):
                        can_extend = False
                        break
                
                if not can_extend:
                    continue
                
                # Get all potential candidates of current_type that neighbor all existing members
                candidates = set()
                for j, (neighbor_type, neighbor_id) in enumerate(self.instance_neighbors.get(instance[0], set())):
                    if neighbor_type == current_type:
                        if j == 0:
                            candidates = {neighbor_id}
                        else:
                            candidates.add(neighbor_id)
                
                # Filter candidates to ensure they neighbor ALL existing members
                for type_id_pair in instance[1:]:
                    current_neighbors = {neighbor_id for neighbor_type, neighbor_id in 
                                         self.instance_neighbors.get(type_id_pair, set()) 
                                         if neighbor_type == current_type}
                    candidates &= current_neighbors
                    if not candidates:
                        break
                
                # Add all valid extensions to new instances
                for candidate_id in candidates:
                    new_instance = instance + ((current_type, candidate_id),)
                    new_instances.append(new_instance)
            
            # If no extensions were found, this pattern has no instances
            if not new_instances:
                return []
            
            current_instances = new_instances
        
        # Extract just the IDs in the right order from the final instances
        result = []
        for instance in current_instances:
            id_list = [id_val for _, id_val in instance]
            result.append(tuple(id_list))
        
        return result

    def get_patterns(self):
        """Returns all discovered patterns sorted by participation index."""
        return sorted(self.patterns, key=lambda p: (-p.pi, len(p.types)))