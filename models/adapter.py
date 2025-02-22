"""
Simple torch module base (or mixin) class providing some commonly used stuff.
"""

from logging import getLogger
from pathlib import Path
import torch
from typing import Any

from common.classes import ClassMap
from common.type_aliases import (
    BatchLabelsT,
    BatchTorchImageT,
    ModelOutputT,
    Number
)


logger = getLogger(__name__)
path = Path(__file__).absolute().parent


class ModuleAdapter(torch.nn.Module):
    __name__ = "NamelessModel"
    root_dir = path

    def __init__(
            self,
            do_input_trafo: bool = False,
            class_map: ClassMap = ClassMap(["unspecified",])
    ) -> None:
        super().__init__()

        self.do_input_trafo = do_input_trafo
        self.class_map = class_map

    def forward(self, x: BatchTorchImageT, *args, **kwargs) -> ModelOutputT:
        # Example:
        # if self.do_input_trafo:
        #     x = type(self).transform_input(x)
        # x = self.model(x)
        # return self.
        raise NotImplementedError()

    @staticmethod
    def transform_input(x: BatchTorchImageT, *args, **kwargs) -> Any:
        """
        Default transformation to take x from torch.Tensor[torch.uint8] with
        shape (b, 3, h, w) and RGB order at the second index to whatever the
        model's forward expects.

        This should be provided as a staticmethod, so that it can be used
        as a torch dataset transformation on loading the data.
        """
        raise NotImplementedError()

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    def training_get_loss(
            self,
            img_batch: BatchTorchImageT,
            labels_batch: BatchLabelsT,
            *args,
            **kwargs
    ) -> tuple[torch.Tensor, dict[str, Number] | None]:
        """
        Compute the scalar loss and possibly return partial losses or any
        kind of info in form of a {"key": scalar_value} dict.
        The dict data will be written to tensorboard.
        Assumes that the model is in train mode.
        """
        # Example:
        # loss_dict = self(img, labels)
        # loss = sum(loss_dict.values())
        # return loss, loss_dict
        raise NotImplementedError()

    def validation_get_loss(
            self,
            img_batch: BatchTorchImageT,
            labels_batch: BatchLabelsT,
            *args,
            **kwargs
    ) -> torch.Tensor:
        """
        Compute the scalar loss. This extra function exists since different
        models could be have different modes of computing the loss, e.g.
        we might have to set the model in training mode.
        Assumes that the model is in eval mode.
        """
        return self.training_get_loss(img_batch, labels_batch, *args, **kwargs)
