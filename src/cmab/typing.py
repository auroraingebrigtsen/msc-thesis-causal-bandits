from typing import Tuple, Dict, TypeAlias

Intervention: TypeAlias = Tuple[str, float]
InterventionSet: TypeAlias = frozenset[Intervention]