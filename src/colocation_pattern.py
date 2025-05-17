class ColocationPattern:
    def __init__(self, types: list, participation_index: float, instances: list):
        self._types = tuple(sorted(types))
        self._pi = participation_index
        self._instances = instances

    def __str__(self) -> str:
        return f"Pattern {self._types} (PI={self._pi:.2f}, Instances={len(self._instances)})"

    def to_dict(self) -> dict:
        return {
            "types": self._types,
            "participation_index": self._pi,
            "num_instances": len(self._instances)
        }

    @property
    def types(self) -> tuple:
        return self._types

    @property
    def pi(self) -> float:
        return self._pi

    @property
    def instances(self) -> list:
        return self._instances
