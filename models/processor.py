from common.type_aliases import BatchPredictionT, BatchTorchImageT, TorchImageT
from logging import getLogger
import torch

from typing import Any

from models.adapter import ModuleAdapter
from utils.bbox_post_processing import Postprocessor


logger = getLogger(__name__)


class ProcessingWrapper(ModuleAdapter):
    def __init__(
            self,
            model: ModuleAdapter,
            mirror_image: bool = False,             # process mirrored image, too
            enlarged_regions_n: int = 0,                    # divide h,w by this and process enlarged regions
            merge_iou_threshold: float = 0.5,       # at which iou to merge bboxes
            pre_merge_score_threshold: float = 0,   # applied before merging algo
            post_merge_score_threshold: float = 0,  # applied to merged bboxes
            merging_method: str = "wbf",
            area_method: str = "int"             # if areas are computed with integer or float coordinates
    ) -> None:
        super().__init__()

        self.model = model
        self.__name__ = f"{model.__name__}Processor"
        self.class_map = model.class_map

        self.mirror_image = mirror_image
        self.enlarged_regions_n = enlarged_regions_n

        self.postprocessor = Postprocessor(
            merge_iou_threshold,
            pre_merge_score_threshold,
            post_merge_score_threshold,
            merging_method,
            area_method
        )

        logger.warning("The ProcessingWrapper has no effect on training!")

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    @staticmethod
    @torch.autograd.grad_mode.inference_mode
    def append_predictions(
            *predictions: BatchPredictionT
    ) -> BatchPredictionT:
        new_dts = []
        for i in range(len(predictions[0])): # batch
            bbs = []
            cls = []
            scs = []

            for dts in predictions:
                bbs.append(dts[i][0])
                cls.append(dts[i][1])
                scs.append(dts[i][2])

            new_dts.append(
                (
                    torch.cat(bbs, dim=0),
                    torch.cat(cls, dim=0),
                    torch.cat(scs, dim=0)
                )
            )
        return new_dts

    @torch.autograd.grad_mode.inference_mode
    def process_mirrored_image(
            self,
            img: BatchTorchImageT
    ) -> BatchPredictionT:
        """
        Mirror an image, predict bounding boxes and mirror them back.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            list: Batch of predictions.
        """
        h, w = img.shape[-2:]

        # Mirror the image horizontally
        img = torch.flip(img, dims=[-1,])  # Flip along width (W)

        # Get detections for both original and mirrored images
        dts = self.model(img)["prediction"]

        # Unmirror the bounding boxes from mirrored detections
        unmirrored_dts = []
        for bbs, cls, scs in dts:
            unmirrored_bbs = torch.clone(bbs)
            unmirrored_bbs[:, 0] = w - bbs[:, 2]
            unmirrored_bbs[:, 2] = w - bbs[:, 0]

            unmirrored_dts.append((unmirrored_bbs, cls, scs))

        return unmirrored_dts

    @torch.autograd.grad_mode.inference_mode
    def process_diced_image(
            self,
            n: int,
            img: BatchTorchImageT
    ) -> BatchPredictionT:
        """
        Processes an image by dividing it into n x n regions, resizing each region to the original size,
        running it through an object detection model, and transforming detections back into original coordinates.

        Args:
            n (int): Number of regions per dimension (image is divided into n x n regions).
            img (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            list: Combined detections from all regions with coordinates transformed to original space.
        """
        H, W = img.shape[-2:]
        region_height = H // n + 1
        region_width = W // n + 1

        # Dice the image into n x n regions
        batched_dts = []
        for i in range(n):
            for j in range(n):
                # Crop the region
                y_start = i * region_height
                y_end = (i + 1) * region_height
                x_start = j * region_width
                x_end = (j + 1) * region_width

                if x_end > W - 1:
                    x_end = W - 1
                    region_width = x_end - x_start
                if y_end > H - 1:
                    y_end = H - 1
                    region_height = y_end - y_start

                region = img[..., y_start:y_end+1, x_start:x_end+1].to(
                    torch.float64
                )

                # Resize the region to the original size
                resized_region = torch.nn.functional.interpolate(
                    region,
                    size=(H, W),
                    mode='bilinear'
                )

                # Run the model on the resized region
                batch_dts = self.model(resized_region)["prediction"]
                new_batch_dts = []
                for bbs, cls, scs in batch_dts:
                    bbs[..., 0] = x_start + bbs[..., 0] * region_width / W
                    bbs[..., 1] = y_start + bbs[..., 1] * region_height / H
                    bbs[..., 2] = x_start + bbs[..., 2] * region_width / W
                    bbs[..., 3] = y_start + bbs[..., 3] * region_height / H
                    new_batch_dts.append((bbs, cls, scs))

                batched_dts.append(new_batch_dts)

        return type(self).append_predictions(*batched_dts)

    def validation_get_loss(self, *args, **kwargs) -> Any:
        return self.model.validation_get_loss(*args, **kwargs)

    def training_get_loss(
            self,
            *args,
            **kwargs
    ) -> Any:
        return self.model.training_get_loss(*args, **kwargs)

    def forward(self, img, *args, **kwargs) -> Any:
        res = self.model(img, *args, **kwargs)

        if self.training:
            return res

        if self.mirror_image:
            preds = self.process_mirrored_image(img)
            res["prediction"] = type(self).append_predictions(
                res["prediction"],
                preds
            )

        if self.enlarged_regions_n > 0:
            preds = self.process_diced_image(self.enlarged_regions_n, img)
            res["prediction"] = type(self).append_predictions(
                res["prediction"],
                preds
            )

        res["prediction"] = self.postprocessor.post_processing(
            res["prediction"]
        )

        return res
