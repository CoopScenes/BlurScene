#!/usr/bin/env python

import os
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import hydra
import logging
import numpy as np
import torch

from omegaconf import OmegaConf
from pathlib import Path

from common.classes import ClassMap
from common.type_aliases import ImageT, PredictionT
from models.processor import ProcessingWrapper
from utils.images import img_to_torch



cfg_path = "config/inference.yaml"

if not Path(cfg_path).exists():
    raise FileNotFoundError(
        f"Inference configuration not found in path {cfg_path}."
    )

cfg = OmegaConf.load(cfg_path)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level = getattr(logging, cfg.logging.level.upper()),
    format = cfg.logging.format
)


def _load_model(
        conf_path: str,
        weights_path: str,
        device: str
) -> tuple[torch.nn.Module, OmegaConf]:
    """
    Get models from model weights checkpoint and the corresponding
    hydra config (usually the one used in training).
    """
    # we load the conf manually, so we don't have to handle all the
    # output hydra would generate with a @hydra.main decorator
    if not Path(conf_path).exists():
        raise FileNotFoundError(
            f"Model configuration not found in path {conf_path}."
        )
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"Model weights not found in path {weights_path}."
        )

    cfg = OmegaConf.load(conf_path)

    model = hydra.utils.instantiate(
        cfg.model.target,
        _convert_="all"
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.eval()

    return model, cfg



class Inference:
    def __init__(self):
        logger.info("Loading face model.")
        self.face_model, self.face_cfg = _load_model(
            cfg.face_model_conf,
            cfg.face_model_weights,
            cfg.device
        )
        logger.info("Loading license plate model.")
        self.lp_model, self.lp_cfg = _load_model(
            cfg.license_plate_model_conf,
            cfg.license_plate_model_weights,
            cfg.device
        )

        if cfg.processing.use:
            kwargs = {k: v for k, v in cfg.processing.items() if k != "use"}
            face_model = ProcessingWrapper(
                model = self.face_model,
                **kwargs
            )
            lp_model = ProcessingWrapper(
                model = self.lp_model,
                **kwargs
            )

        self.face_model.to(cfg.device)
        self.lp_model.to(cfg.device)

        self.class_map = ClassMap(["face", "license plate"])

        logger.info("Preparing image normalization pipeline.")
        if (
                self.face_cfg.default_trafo != self.lp_cfg.default_trafo or
                self.face_cfg.image_width != self.lp_cfg.image_width or
                self.face_cfg.image_height != self.lp_cfg.image_height
        ):
            raise NotImplementedError(
                "Image transformations for face and license plate models differ."
                "Only both models using the same transformation is implemented."
            )

        self.preprocessing_trafo = hydra.utils.instantiate(
            self.face_cfg.default_trafo,
            _convert_="all"
        )

        # warmup/compile models
        logger.info("Model warmup. This can take a while if the model is compiled.")
        dummy_img = np.random.randint(
            0,
            255,
            (self.face_cfg.image_height, self.face_cfg.image_width, 3),
            dtype=np.uint8
        )
        dummy_res = self.predict(dummy_img)
        del dummy_img
        del dummy_res

        logger.info("Models ready.")


    @torch.autograd.grad_mode.inference_mode()
    def predict(self, img: ImageT) -> PredictionT:
        # if the preprocessing_trafo rescales the image in any way,
        # we use this dummy box to compute the reverse coordinate trafo for
        # predicted bboxes
        h, w = img.shape[:2]
        dummy_orig = np.array([[0, 0, w, h, 0],])

        # preprocess, e.g. CLAHE, resize, pad, to torch.Tensor, add batch dimension
        preproc = self.preprocessing_trafo(image=img, bboxes=dummy_orig)
        img = img_to_torch(preproc["image"])[None, ...]
        img = img.to(cfg.device)

        with torch.autocast(cfg.device, enabled=self.face_cfg.with_amp):
            face_boxes, face_class, face_scores = self.face_model(img)["prediction"][0]

        with torch.autocast(cfg.device, enabled=self.lp_cfg.with_amp):
            lp_boxes, lp_class, lp_scores = self.lp_model(img)["prediction"][0]

        # fix classes since models only have 1 class, i.e. face_class and
        # lp_class are zero
        face_class[:] = self.class_map.name_to_index["face"]
        lp_class[:] = self.class_map.name_to_index["license plate"]

        boxes = torch.cat([face_boxes, lp_boxes])
        classes = torch.cat([face_class, lp_class])
        scores = torch.cat([face_scores, lp_scores])

        # transform predicted bboxes to original coordinates
        dummy_trafo = preproc["bboxes"]
        dx = dummy_orig[0, 0] - dummy_trafo[0, 0]
        dy = dummy_orig[0, 1] - dummy_trafo[0, 1]
        scale_x = w / (dummy_trafo[0, 2] - dummy_trafo[0, 0])
        scale_y = h / (dummy_trafo[0, 3] - dummy_trafo[0, 1])

        boxes[:, 0] = (boxes[:, 0] + dx) * scale_x
        boxes[:, 2] = (boxes[:, 2] + dx) * scale_x
        boxes[:, 1] = (boxes[:, 1] + dy) * scale_y
        boxes[:, 3] = (boxes[:, 3] + dy) * scale_y

        # TODO to cpu? to numpy? separate lp and face results?

        return boxes, classes, scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        "Execution of this code returns the detected bounding boxes "
        "with class and score."
    )
    parser.add_argument(
        "image",
        help="Image to be searched for faces and license plates"
    )
    args = parser.parse_args()

    from utils.images import read_img
    img = read_img(args.image)

    ifrc = Inference()
    bbs, cls, scs = ifrc.predict(img)
    bbs = bbs.to("cpu")
    cls = cls.to("cpu")
    scs = scs.to("cpu")

    for b, c, s in zip(bbs, cls, scs):
        print(
            f"[{int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])}, "
            f"{int(c)}, {float(s):.2f}]"
        )
