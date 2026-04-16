"""
Some custom type to be reused for type annotations.
Serves mainly to keep track what default inputs and outputs our models should
have.
"""

from typing import Optional, Sequence, Tuple, TypedDict, Union

from numpy import float32, uint8, number
from numpy.typing import NDArray
from torch import Tensor

## Image types

# our default image type is (h, w, 3)
ImageT = NDArray[uint8]
FloatImageT = NDArray[float32]

# image returned by our datasets is the default torch tensor format (3, h, w)
# the element type can't be annotated; it should be torch.Tensor[torch.uint8]
TorchImageT = Tensor

# batched torch images, again torch.Tensor[torch.uint8] with extra outer dim
# (b, 3, h, w)
BatchTorchImageT = Tensor


## Bounding Boxes, Ground Truths and Detections

# generic tuple with x0, y0, x1, y1 and possibly clsindex, and maybe score
Number = Union[int, float, number]
GenericBboxesT = Union[Sequence[Sequence[Number]], NDArray, Tensor]

# used from dataset to model output
BboxesT = Tensor  # coordinates of bounding boxes, float, shape (n, 4)
ScoresT = Tensor  # scores, float in (0,1), shape (n,)
ClassesT = Tensor  # bbox classindices, int, shape (n,)

LabelsT = Tuple[BboxesT, ClassesT]
BatchLabelsT = Sequence[LabelsT]

# each dataset returns an image and a bboxlist of bboxes w. classes in the image
DataItemT = Tuple[
    TorchImageT,
    Optional[LabelsT],
]
# a dataloader batches these together
BatchDataItemT = Tuple[
    BatchTorchImageT,
    Union[BatchLabelsT, Sequence[None]],
]

# if we need to know the index of an image in a dataset
IndexT = int
LabelsWIndexT = Union[Tuple[BboxesT, ClassesT, IndexT], IndexT]
BatchLabelsWIndexT = Sequence[LabelsWIndexT]
DataItemWIndexT = Tuple[TorchImageT, LabelsWIndexT]
BatchDataItemWIndexT = Tuple[BatchTorchImageT, BatchLabelsWIndexT]


## model outputs

# model's forward function in eval mode should return a 3-tuple of tensors.
# The tensors should hold the bounding box coordinates, class indices and scores.
PredictionT = Tuple[BboxesT, ClassesT, ScoresT]
BatchPredictionT = Sequence[PredictionT]

# since model in train mode might have different output, there might be the
# need to return more, so the model output is wrapped in a dict.
ModelOutputT = TypedDict(
    "ModelOutputT",
    {
        "prediction": BatchPredictionT,  # must have in eval mode
        "loss": float,  # in training mode
    },
    total=False,
)
