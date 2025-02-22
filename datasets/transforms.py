import albumentations as A
from albumentations.core.types import Targets
from albumentations.core.bbox_utils import normalize_bboxes
from albumentations.augmentations.geometric.functional import (
    scale,
    pad_with_params,
    pad_bboxes,
    denormalize_bboxes
)
import cv2
from typing import Any, Sequence

from common.type_aliases import GenericBboxesT, ImageT


class PaddedResize(A.DualTransform):
    """
    Resize an image and bounding boxes, such that the aspect ratio of the image
    is preserved and pad the resized image so that the target shape is filled.
    I.e. returns a resized image centered in an e.g. white background with the
    target shape.
    """
    # albumentations trafo interface in a nutshell:
    # albumentations defines in the BasicTransform class where parameters
    # for the transformation of all targets are computed:
    # get_params_dependent_on_data will be called with a params dict, which holds
    # the image shape, and the transform targets, e.g. image and bboxes.
    # the return value has to be a dict as well, which will then be merged with
    # params. the updated params then get passed on to the different apply*
    # methods.
    # the bounding box coordinates are already normalized to [0,1]
    #
    # parts of the code are directly copied from the albumentations source.
    _targets = (Targets.IMAGE, Targets.BBOXES)

    def __init__(
            self,
            width: int,
            height: int,
            pad_value: int | Sequence[int] = 0,
            interpolation: int = cv2.INTER_LINEAR,
            border_mode: int = cv2.BORDER_CONSTANT,
            p: float = 1.0
    ) -> None:
        super().__init__(p=p)

        self.width = width
        self.height = height
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.border_mode = border_mode

        self.target_aspect_ratio = height / width

    def get_transform_init_args_names(self) -> tuple[str]:
        return (
            "width",
            "height",
            "pad_value",
            "interpolation",
            "border_mode",
            "p"
        )

    def get_params_dependent_on_data(
            self,
            params: dict[str, Any],
            data: dict[str, Any],
    ) -> dict[str, int]:
        """
        Compute the scale factors and coordinate offsets.
        """
        h_img, w_img  = params["shape"][:2]
        aspect_ratio = h_img / w_img

        # resize scale
        if aspect_ratio >= self.target_aspect_ratio:
            scale_factor = self.height / h_img
        else:
            scale_factor = self.width / w_img

        scaled_width = int(scale_factor * w_img)
        scaled_height = int(scale_factor * h_img)

        # padding
        if scaled_height < self.height:
            pad_top = int((self.height - scaled_height) / 2.0)
            pad_bottom = self.height - scaled_height - pad_top
        else:
            pad_top = 0
            pad_bottom = 0

        if scaled_width < self.width:
            pad_left = int((self.width - scaled_width) / 2.0)
            pad_right = self.width - scaled_width - pad_left
        else:
            pad_left = 0
            pad_right = 0

        return {
            "scale_factor": scale_factor,
            "scaled_width": scaled_width,
            "scaled_height": scaled_height,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right
        }

    def apply(
            self,
            img: ImageT,
            scale_factor: float,
            pad_top: int,
            pad_bottom: int,
            pad_left: int,
            pad_right: int,
            **params: Any
    ) -> ImageT:
        """
        Rescale and pad the image.
        """
        img = scale(
            img,
            scale_factor,
            self.interpolation
        )

        return pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.pad_value,
        )

    def apply_to_bboxes(
            self,
            bboxes: GenericBboxesT,
            scaled_width: int,
            scaled_height: int,
            pad_top: int,
            pad_bottom: int,
            pad_left: int,
            pad_right: int,
            **params: Any
    ) -> GenericBboxesT:
        """
        Adjust bboxes for padding.
        """
        scaled_img_shape = [scaled_height, scaled_width, *(params["shape"][2:])]
        bboxes_np = denormalize_bboxes(bboxes, scaled_img_shape)

        result = pad_bboxes(
            bboxes_np,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            self.border_mode,
            image_shape=scaled_img_shape,
        )

        return normalize_bboxes(
            result,
            (
                scaled_height + pad_top + pad_bottom,
                scaled_width + pad_left + pad_right
            )
        )
