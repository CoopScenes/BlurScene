"""
Some custom type to be reused for type annotations.
Serves mainly to keep track what default inputs and outputs our models should
have.
"""

from collections.abc import Sequence
from numpy import float32, uint8, number
from numpy.typing import NDArray
from torch import Tensor
from typing import TypeAlias, TypedDict

## Image types

# our default image type is (h, w, 3)
ImageT: TypeAlias = NDArray[uint8]
FloatImageT: TypeAlias = NDArray[float32]

# image returned by our datasets is the default torch tensor format (3, h, w)
# the element type can't be annotated; it should be torch.Tensor[torch.uint8]
TorchImageT: TypeAlias = Tensor

# batched torch images, again torch.Tensor[torch.uint8] with extra outer dim
# (b, 3, h, w)
BatchTorchImageT: TypeAlias = Tensor


## Bounding Boxes, Ground Truths and Detections

# generic tuple with x0, y0, x1, y1 and possibly clsindex, and maybe score
Number: TypeAlias = int | float | number
GenericBboxesT = Sequence[Sequence[Number]] | NDArray | Tensor

# used from dataset to model output
BboxesT: TypeAlias = Tensor # coordinates of bounding boxes, float, shape (n, 4)
ScoresT: TypeAlias = Tensor # scores, float in (0,1), shape (n,)
ClassesT: TypeAlias = Tensor # bbox classindices, int, shape (n,)

LabelsT: TypeAlias = tuple[BboxesT, ClassesT]
BatchLabelsT: TypeAlias = Sequence[LabelsT]

# each dataset returns an image and a bboxlist of bboxes w. classes in the image
DataItemT: TypeAlias = tuple[
    TorchImageT,
    LabelsT | None
]
# a dataloader batches these together
BatchDataItemT: TypeAlias = tuple[
    BatchTorchImageT,
    BatchLabelsT | Sequence[None]
]

# if we need to know the index of an image in a dataset
IndexT: TypeAlias = int
LabelsWIndexT: TypeAlias = tuple[BboxesT, ClassesT, IndexT] | IndexT
BatchLabelsWIndexT: TypeAlias = Sequence[LabelsWIndexT]
DataItemWIndexT: TypeAlias = tuple[TorchImageT, LabelsWIndexT]
BatchDataItemWIndexT:  TypeAlias = tuple[BatchTorchImageT, BatchLabelsWIndexT]


## model outputs

# model's forward function in eval mode should return a 3-tuple of tensors.
# The tensors should hold the bounding box coordinates, class indices and scores.
PredictionT: TypeAlias = tuple[BboxesT, ClassesT, ScoresT]
BatchPredictionT: TypeAlias = Sequence[PredictionT]

# since model in train mode might have different output, there might be the
# need to return more, so the model output is wrapped in a dict.
ModelOutputT: TypeAlias = TypedDict(
    "ModelOutputT",
    {
        "prediction": BatchPredictionT,     # must have in eval mode
        "loss": float     # in training mode
    }
)
