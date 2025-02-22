from collections.abc import Sequence
from collections import defaultdict


class ClassMap():
    """
    Takes a list of classes and deterministically generates a map between class
    names and consecutive indices. The map is unique to a set of classes, i.e.
    the order of the names in the init list is not relevant.
    Non-existing class names are mapped to None.
    """
    def __init__(self, classes: Sequence[str]):
        self.classes: tuple[str, ...] = tuple(sorted(set(classes)))

        self.index_to_name: defaultdict[int, str | None] = defaultdict(lambda: None)
        self.name_to_index: defaultdict[str, int | None] = defaultdict(lambda: None)

        for idx, name in enumerate(self.classes):
            self.index_to_name[idx] = name
            self.name_to_index[name] = idx

        self.num_classes: int = len(self.index_to_name)

    def __len__(self) -> int:
        return len(self.index_to_name)

    def __contains__(self, x: str) -> bool:
        return x in self.classes
