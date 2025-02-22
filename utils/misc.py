import numpy as np
import h5py

from collections.abc import Sequence

from common.type_aliases import (
    BatchLabelsT,
    BatchPredictionT,
    LabelsT,
    PredictionT
)


def labels_to_device(
        labels: BatchLabelsT | BatchPredictionT,
        device: str,
        non_blocking: bool = False
) -> BatchLabelsT | BatchPredictionT:
    return [
        tuple(x.to(device, non_blocking=non_blocking) for x in y)
        for y in labels
    ]


def label_to_device(
        label: LabelsT | PredictionT,
        device: str,
        non_blocking: bool = False
) -> LabelsT | PredictionT:
    return tuple(x.to(device, non_blocking=non_blocking) for x in label)


def save_predictions_hdf(
        filename: str,
        data_ids: Sequence[np.typing.NDArray], # each item is a prediction for a specific image
        bboxes: Sequence[np.typing.NDArray],   # i.e. bboxes[idx] holds all bboxes for the
        classes: Sequence[np.typing.NDArray],  # i'th image of the dataset
        scores: Sequence[np.typing.NDArray],
):
    d = {
        "data_ids": data_ids,
        "bboxes": bboxes,
        "classes": classes,
        "scores": scores
    }
    with h5py.File(filename, "w") as hdf:
        for name, data in d.items():
            g = hdf.create_group(name, track_order=True)
            for i, array in enumerate(data):
                g.create_dataset(str(i), data=array)


def load_predictions_hdf(
        filename: str
) -> tuple[
    list[np.typing.NDArray],
    list[np.typing.NDArray],
    list[np.typing.NDArray],
    list[np.typing.NDArray]
]:
    d = {
        "data_ids": None,
        "bboxes": None,
        "classes": None,
        "scores": None
    }

    with h5py.File(filename, "r") as hdf:
        for name, group in hdf.items():
            d[name] = [x[...] for x in group.values()]

    return d
