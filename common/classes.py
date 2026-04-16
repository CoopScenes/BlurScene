from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Optional, Sequence, Tuple


class ClassMap():
    """
    Takes a list of classes and deterministically generates a map between class
    names and consecutive indices. The map is unique to a set of classes, i.e.
    the order of the names in the init list is not relevant.
    Non-existing class names are mapped to None.
    """
    def __init__(self, classes: Sequence[str]):
        self.classes: Tuple[str, ...] = tuple(sorted(set(classes)))

        self.index_to_name: DefaultDict[int, Optional[str]] = defaultdict(lambda: None)
        self.name_to_index: DefaultDict[str, Optional[int]] = defaultdict(lambda: None)

        for idx, name in enumerate(self.classes):
            self.index_to_name[idx] = name
            self.name_to_index[name] = idx

        self.num_classes: int = len(self.index_to_name)

    def __len__(self) -> int:
        return len(self.index_to_name)

    def __contains__(self, x: str) -> bool:
        return x in self.classes
