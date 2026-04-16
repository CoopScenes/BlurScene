from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from common.type_aliases import GenericBboxesT, ImageT


def _to_plain_value(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


class BboxParams:
    def __init__(
            self,
            format: str = "pascal_voc",
            min_area: float = 0,
            min_visibility: float = 0
    ) -> None:
        self.format = format
        self.min_area = min_area
        self.min_visibility = min_visibility


class Compose:
    def __init__(
            self,
            transforms: Sequence,
            bbox_params: Optional[BboxParams] = None,
            p: float = 1.0
    ) -> None:
        self.transforms = list(transforms)
        self.bbox_params = bbox_params
        self.p = p

    def __call__(self, image: ImageT, bboxes: GenericBboxesT):
        result = {
            "image": image,
            "bboxes": np.array(bboxes, dtype=np.float32),
        }
        for transform in self.transforms:
            result = transform(image=result["image"], bboxes=result["bboxes"])
        return result


class CLAHE:
    def __init__(
            self,
            clip_limit: float = 4.0,
            tile_grid_size: Sequence[int] = (8, 8),
            p: float = 1.0
    ) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)
        self.p = p

    def __call__(self, image: ImageT, bboxes: GenericBboxesT):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return {
            "image": cv2.cvtColor(merged, cv2.COLOR_LAB2RGB),
            "bboxes": np.array(bboxes, dtype=np.float32),
        }


class PaddedResize:
    """
    Resize an image and bounding boxes while preserving aspect ratio and pad to
    the target size.
    """
    def __init__(
            self,
            width: int,
            height: int,
            pad_value: int | Sequence[int] = 0,
            interpolation: int = cv2.INTER_LINEAR,
            border_mode: int = cv2.BORDER_CONSTANT,
            p: float = 1.0
    ) -> None:
        self.width = width
        self.height = height
        self.pad_value = _to_plain_value(pad_value)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.p = p
        self.target_aspect_ratio = height / width

    def __call__(self, image: ImageT, bboxes: GenericBboxesT):
        bboxes_np = np.array(bboxes, dtype=np.float32).copy()
        h_img, w_img = image.shape[:2]
        aspect_ratio = h_img / w_img

        if aspect_ratio >= self.target_aspect_ratio:
            scale_factor = self.height / h_img
        else:
            scale_factor = self.width / w_img

        scaled_width = int(scale_factor * w_img)
        scaled_height = int(scale_factor * h_img)
        resized = cv2.resize(
            image,
            (scaled_width, scaled_height),
            interpolation=self.interpolation,
        )

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

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=self.border_mode,
            value=self.pad_value,
        )

        if bboxes_np.size > 0:
            bboxes_np[:, 0] = bboxes_np[:, 0] * scale_factor + pad_left
            bboxes_np[:, 2] = bboxes_np[:, 2] * scale_factor + pad_left
            bboxes_np[:, 1] = bboxes_np[:, 1] * scale_factor + pad_top
            bboxes_np[:, 3] = bboxes_np[:, 3] * scale_factor + pad_top

        return {
            "image": padded,
            "bboxes": bboxes_np,
        }


def build_from_config(conf) -> Compose:
    transforms_list: List = []
    for transform_conf in conf.transforms:
        target = transform_conf["_target_"]
        if target == "albumentations.CLAHE":
            transforms_list.append(
                CLAHE(
                    clip_limit=transform_conf.get("clip_limit", 4.0),
                    tile_grid_size=transform_conf.get("tile_grid_size", (8, 8)),
                    p=transform_conf.get("p", 1.0),
                )
            )
        elif target == "datasets.transforms.PaddedResize":
            border_mode = _to_plain_value(transform_conf.get("border_mode"))
            if isinstance(border_mode, dict) and border_mode.get("_target_") == "hydra.utils.get_object":
                if border_mode.get("path") == "cv2.BORDER_CONSTANT":
                    border_mode = cv2.BORDER_CONSTANT
            border_mode = int(border_mode if border_mode is not None else cv2.BORDER_CONSTANT)
            transforms_list.append(
                PaddedResize(
                    width=transform_conf.width,
                    height=transform_conf.height,
                    pad_value=_to_plain_value(transform_conf.get("pad_value", 0)),
                    interpolation=int(_to_plain_value(transform_conf.get("interpolation", cv2.INTER_LINEAR))),
                    border_mode=border_mode,
                    p=transform_conf.get("p", 1.0),
                )
            )
        else:
            raise NotImplementedError(f"Unsupported preprocessing transform: {target}")

    bbox_params = None
    if conf.get("bbox_params") is not None:
        bbox_params = BboxParams(
            format=conf.bbox_params.get("format", "pascal_voc"),
            min_area=conf.bbox_params.get("min_area", 0),
            min_visibility=conf.bbox_params.get("min_visibility", 0),
        )

    return Compose(
        transforms=transforms_list,
        bbox_params=bbox_params,
        p=conf.get("p", 1.0),
    )
