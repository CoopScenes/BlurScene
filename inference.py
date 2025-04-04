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
from typing import List, Tuple

from common.classes import ClassMap
from common.type_aliases import ImageT, PredictionT
from models.processor import ProcessingWrapper
from utils.images import img_to_torch

cfg_path = "config/inference.yaml"

if not Path(cfg_path).exists():
    raise FileNotFoundError(f"Inference configuration not found in path {cfg_path}.")

cfg = OmegaConf.load(cfg_path)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, cfg.logging.level.upper()),
    format=cfg.logging.format
)

def _load_model(conf_path: str, weights_path: str, device: str) -> Tuple[torch.nn.Module, OmegaConf]:
    """
    Lädt ein Modell aus den angegebenen Konfigurations- und Gewichtedateien und verschiebt es auf das gewünschte Gerät.
    """
    if not Path(conf_path).exists():
        raise FileNotFoundError(f"Model configuration not found in path {conf_path}.")
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Model weights not found in path {weights_path}.")

    model_cfg = OmegaConf.load(conf_path)
    model = hydra.utils.instantiate(model_cfg.model.target, _convert_="all")
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.eval()
    return model, model_cfg

class Inference:
    def __init__(self):
        """
        Initialisiert die Inferenzklasse, lädt beide Modelle auf unterschiedliche GPUs und bereitet das Preprocessing vor.
        """
        # Multi-GPU: Neue Config-Einträge für explizite Gerätezuordnung
        self.face_device = cfg.get("face_device", cfg.device)           # z. B. "cuda:0"
        self.lp_device = cfg.get("license_plate_device", cfg.device)      # z. B. "cuda:1"

        logger.info("Loading face model.")
        self.face_model, self.face_cfg = _load_model(
            cfg.face_model_conf,
            cfg.face_model_weights,
            self.face_device
        )
        logger.info("Loading license plate model.")
        self.lp_model, self.lp_cfg = _load_model(
            cfg.license_plate_model_conf,
            cfg.license_plate_model_weights,
            self.lp_device
        )

        # Optionale Pre-/Post-Processing Wrapper
        if cfg.processing.use:
            kwargs = {k: v for k, v in cfg.processing.items() if k != "use"}
            self.face_model = ProcessingWrapper(model=self.face_model, **kwargs)
            self.lp_model = ProcessingWrapper(model=self.lp_model, **kwargs)

        self.face_model.to(self.face_device)
        self.lp_model.to(self.lp_device)

        self.class_map = ClassMap(["face", "license plate"])

        logger.info("Preparing image normalization pipeline.")
        if (
            self.face_cfg.default_trafo != self.lp_cfg.default_trafo or
            self.face_cfg.image_width != self.lp_cfg.image_width or
            self.face_cfg.image_height != self.lp_cfg.image_height
        ):
            raise NotImplementedError(
                "Image transformations for face and license plate models differ. "
                "Only both models using the same transformation is implemented."
            )

        self.preprocessing_trafo = hydra.utils.instantiate(
            self.face_cfg.default_trafo,
            _convert_="all"
        )

        # Warmup der Modelle (Einzelbild-Warmup reicht als Demo)
        logger.info("Model warmup. This can take a while if the model has to be compiled.")
        dummy_img = np.random.randint(
            0,
            255,
            (self.face_cfg.image_height, self.face_cfg.image_width, 3),
            dtype=np.uint8
        )
        _ = self.predict(dummy_img)
        logger.info("Models ready.")

    @torch.autograd.grad_mode.inference_mode()
    def predict(self, img: ImageT) -> PredictionT:
        """
        Führt die Inferenz für ein einzelnes Bild durch.
        """
        h, w = img.shape[:2]
        dummy_orig = np.array([[0, 0, w, h, 0]], dtype=np.float32)

        preproc = self.preprocessing_trafo(image=img, bboxes=dummy_orig)
        img_tensor = img_to_torch(preproc["image"])[None, ...]
        img_tensor = img_tensor.to(cfg.device)

        with torch.autocast(self.face_device, enabled=self.face_cfg.with_amp):
            face_boxes, face_class, face_scores = self.face_model(img_tensor.to(self.face_device))["prediction"][0]

        with torch.autocast(self.lp_device, enabled=self.lp_cfg.with_amp):
            lp_boxes, lp_class, lp_scores = self.lp_model(img_tensor.to(self.lp_device))["prediction"][0]

        face_class[:] = self.class_map.name_to_index["face"]
        lp_class[:] = self.class_map.name_to_index["license plate"]

        boxes = torch.cat([face_boxes, lp_boxes])
        classes = torch.cat([face_class, lp_class])
        scores = torch.cat([face_scores, lp_scores])

        dummy_trafo = preproc["bboxes"]
        dx = dummy_orig[0, 0] - dummy_trafo[0, 0]
        dy = dummy_orig[0, 1] - dummy_trafo[0, 1]
        scale_x = w / (dummy_trafo[0, 2] - dummy_trafo[0, 0])
        scale_y = h / (dummy_trafo[0, 3] - dummy_trafo[0, 1])

        boxes[:, 0] = (boxes[:, 0] + dx) * scale_x
        boxes[:, 2] = (boxes[:, 2] + dx) * scale_x
        boxes[:, 1] = (boxes[:, 1] + dy) * scale_y
        boxes[:, 3] = (boxes[:, 3] + dy) * scale_y

        return boxes, classes, scores

    @torch.autograd.grad_mode.inference_mode()
    def predict_batch(self, imgs: List[ImageT]) -> List[PredictionT]:
        """
        Führt die Inferenz für einen Batch von Bildern durch.
        Erwartet, dass alle Bilder ähnliche Dimensionen haben.
        """
        processed_imgs = []
        dummy_orig_list = []
        trafo_list = []
        sizes = []
        for img in imgs:
            h, w = img.shape[:2]
            sizes.append((h, w))
            dummy_orig = np.array([[0, 0, w, h, 0]], dtype=np.float32)
            dummy_orig_list.append(dummy_orig)
            preproc = self.preprocessing_trafo(image=img, bboxes=dummy_orig)
            processed_imgs.append(img_to_torch(preproc["image"]))
            trafo_list.append(preproc["bboxes"])
        batch = torch.stack(processed_imgs)  # Shape: (B, C, H, W)
        B = batch.shape[0]

        # Gesichtsmodell auf face_device
        batch_face = batch.to(self.face_device)
        with torch.autocast(self.face_device, enabled=self.face_cfg.with_amp):
            face_preds = self.face_model(batch_face)["prediction"]
        # Kennzeichenmodell auf lp_device
        batch_lp = batch.to(self.lp_device)
        with torch.autocast(self.lp_device, enabled=self.lp_cfg.with_amp):
            lp_preds = self.lp_model(batch_lp)["prediction"]

        results = []
        for i in range(B):
            face_boxes, face_class, face_scores = face_preds[i]
            lp_boxes, lp_class, lp_scores = lp_preds[i]

            face_class[:] = self.class_map.name_to_index["face"]
            lp_class[:] = self.class_map.name_to_index["license plate"]

            boxes = torch.cat([face_boxes, lp_boxes])
            classes = torch.cat([face_class, lp_class])
            scores = torch.cat([face_scores, lp_scores])

            dummy_orig = dummy_orig_list[i]
            trafo = trafo_list[i]
            h, w = sizes[i]
            dx = dummy_orig[0, 0] - trafo[0, 0]
            dy = dummy_orig[0, 1] - trafo[0, 1]
            scale_x = w / (trafo[0, 2] - trafo[0, 0])
            scale_y = h / (trafo[0, 3] - trafo[0, 1])
            boxes[:, 0] = (boxes[:, 0] + dx) * scale_x
            boxes[:, 2] = (boxes[:, 2] + dx) * scale_x
            boxes[:, 1] = (boxes[:, 1] + dy) * scale_y
            boxes[:, 3] = (boxes[:, 3] + dy) * scale_y

            results.append((boxes, classes, scores))
        return results

if __name__ == "__main__":
    import argparse
    from utils.images import read_img

    parser = argparse.ArgumentParser(
        "Führt die Inferenz aus und gibt die erkannten Bounding Boxes mit Klassen und Scores aus."
    )
    parser.add_argument("image", help="Pfad zum zu verarbeitenden Bild")
    args = parser.parse_args()

    img = read_img(args.image)
    inference = Inference()
    bbs, cls, scs = inference.predict(img)
    bbs = bbs.to("cpu")
    cls = cls.to("cpu")
    scs = scs.to("cpu")

    for b, c, s in zip(bbs, cls, scs):
        print(f"[{int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])}, {int(c)}, {float(s):.2f}]")
