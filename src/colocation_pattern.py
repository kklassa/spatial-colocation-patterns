from typing import Tuple, Dict, List

from src.types import FeatureType, PatternInstance


class ColocationPattern:
    def __init__(
        self, 
        types: Tuple[FeatureType], 
        participation_index: float, 
        instances: List[PatternInstance],
    ):
        self._types = tuple(sorted(types))
        self._pi = participation_index
        self._instances = instances

    def __str__(self) -> str:
        return f"Pattern {self._types} (PI={self._pi:.2f}, Instances={len(self._instances)})"

    def to_dict(self) -> Dict[str, object]:
        """
        Converts the colocation pattern to a dictionary representation.

        Returns:
            A dictionary with the types, participation index, and number of instances.
        """

        return {
            "types": self._types,
            "participation_index": self._pi,
            "num_instances": len(self._instances)
        }

    @property
    def types(self) -> Tuple[FeatureType]:
        """
        Returns the types of the colocation pattern.

        Returns:
            A tuple of types in the colocation pattern.
        """
        return self._types

    @property
    def pi(self) -> float:
        """
        Returns the participation index of the colocation pattern.

        Returns:
            A float representing the participation index.
        """
        return self._pi

    @property
    def instances(self) -> List[PatternInstance]:
        """
        Returns the instances of the colocation pattern.

        Returns:
            A list of instances in the colocation pattern.
        """
        return self._instances
