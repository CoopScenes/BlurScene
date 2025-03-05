import torch
import torchvision.models.detection.faster_rcnn as tmdf
import torchvision.ops.poolers as top

from collections.abc import Callable
from logging import getLogger
from math import floor, sqrt
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.ops import complete_box_iou_loss, sigmoid_focal_loss
from torchvision.ops._utils import _upcast
from torchvision.ops.boxes import box_area, box_iou
from torchvision.utils import _log_api_usage_once
from typing import Optional

from common.classes import ClassMap
from common.type_aliases import (
    BatchLabelsT,
    BatchTorchImageT,
    ClassesT,
    ModelOutputT,
    Number,
    Sequence
)
from models.adapter import ModuleAdapter


logger = getLogger(__name__)


class FasterRCNNWrapper(ModuleAdapter):
    __name__ = "FasterRCNNWrapper"

    def __init__(
            self,
            model: tmdf.FasterRCNN,
            class_map: ClassMap = ClassMap(["unspecified",]),
            do_input_trafo: bool = True,
            mult_obj_loss: float = 1.0,
            mult_rpn_reg_loss: float = 1.0,
            mult_cls_loss: float = 1.0,
            mult_box_reg_loss: float = 1.0,
            compile_parts: bool = False
    ) -> None:
        super().__init__(
            do_input_trafo=do_input_trafo,
            class_map=class_map
        )

        self.model = model

        self.mult_obj_loss = mult_obj_loss
        self.mult_rpn_reg_loss = mult_rpn_reg_loss
        self.mult_cls_loss = mult_cls_loss
        self.mult_box_reg_loss = mult_box_reg_loss

        if compile_parts:
            self.compile_fixed_shape_submodules(dynamic=False)

    @staticmethod
    def transform_input(x: BatchTorchImageT) -> BatchTorchImageT:
        return x.to(torch.float32) / 255.

    def training_get_loss(
            self,
            img: BatchTorchImageT,
            labels: BatchLabelsT
    ) -> tuple[torch.Tensor, dict[str, Number] | None]:
        # here we simply need to reduce the dictionary returned by faster r-cnn
        # to a scalar, i.e. we sum the different losses returned by the
        # net in training mode
        loss_dict = self(img, labels)
        loss_obj = loss_dict["loss_objectness"]
        loss_rpn_reg = loss_dict["loss_rpn_box_reg"]
        loss_cls = loss_dict["loss_classifier"]
        loss_box_reg = loss_dict["loss_box_reg"]

        loss = self.mult_obj_loss * loss_obj + \
            self.mult_rpn_reg_loss * loss_rpn_reg + \
            self.mult_cls_loss * loss_cls + \
            self.mult_box_reg_loss * loss_box_reg

        return loss, loss_dict

    def validation_get_loss(
            self,
            img: BatchTorchImageT,
            labels: BatchLabelsT
    ) -> torch.Tensor:
        prev_mode = self.training
        self.partial_eval_mode()

        loss, _ = self.training_get_loss(img, labels)

        self.train(prev_mode)

        return loss

    def partial_eval_mode(self) -> None:
        self.train(True)
        for _, m in self.named_modules(remove_duplicate=False):
            name = str(type(self)).lower()
            if "batchnorm" in name or "dropout" in name:
                m.train(False)
            else:
                m.train(True)

    def compile_fixed_shape_submodules(self, **kwargs):
        """
        Tries to set the parts of the model that will not have varying
        input and output shapes (if using the same shape of batches of images)
        to JIT.

        Large parts of the model have varying in- and outputs, in training due
        to using the image-dependent ground truths, and in inference due
        to the image-dependent number of predictions.
        """
        logger.debug("Replacing parts of the model with JIT versions.")
        # complete fpn should be fixed in and output if img input shape is fixed
        self.model.backbone = torch.compile(
            self.model.backbone,
            **kwargs
        )

        # rpn has some fixed shapes, some dynamic:
        # in training: input ground truths
        # in inference: output bbox proposals
        self.model.rpn.anchor_generator = torch.compile(
            self.model.rpn.anchor_generator,
            **kwargs
        )
        self.model.rpn.head = torch.compile(
            self.model.rpn.head,
            **kwargs
        )
        self.model.rpn.box_coder.decode = torch.compile(
            self.model.rpn.box_coder.decode,
            **kwargs
        )

    def forward(
            self,
            x: BatchTorchImageT,
            targets: BatchLabelsT | Sequence[None] | None = None
    ) -> ModelOutputT:
        if self.do_input_trafo:
            x = type(self).transform_input(x)

        if not self.training:
            results = self.model(x)

            return {
                "prediction": [
                    (
                        r["boxes"],
                        r["labels"] - 1, # remove additional background class 0
                        r["scores"]
                    )
                    for r in results
                ]
            }

        targets = [
            {
                "boxes": bboxes,
                "labels": classes + 1 # add background class 0
            }
            for bboxes, classes in targets
        ]
        results = self.model(x, targets)
        # {
        #     'loss_classifier': tensor
        #     'loss_box_reg': tensor
        #     'loss_objectness': tensor
        #     'loss_rpn_box_reg': tensor
        # }
        return results


###
### monkey patches / some replacements for torchvision functions
###

# #
# # RegionProposalNetwork with DIoU and Focal Losses
# #

# OriginalRegionProposalNetwork = tmdf.RegionProposalNetwork

# class RegionProposalNetwork(OriginalRegionProposalNetwork):
#     def compute_loss(
#             self,
#             objectness: torch.Tensor,
#             proposals: torch.Tensor, # [B,N,4]
#             labels: list[torch.Tensor], # list is batch dimension, tensor [N_B,]
#             gt_bboxes: list[torch.Tensor] # list is batch dimension, tensor [N_B,4]
#         ) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         torchvision.models.detection.rpn.compute_loss with DIoU and focal losses.
#         """
#         sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
#         sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
#         sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

#         sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

#         objectness = objectness.flatten()

#         labels = torch.cat(labels, dim=0)
#         gt_bboxes = torch.cat(gt_bboxes, dim=0)
#         proposals = proposals.view(-1, 4)

#         box_loss = complete_box_iou_loss(
#             proposals[sampled_pos_inds],
#             gt_bboxes[sampled_pos_inds],
#             reduction="sum"
#         ) / (sampled_inds.numel())
#         # originally:
#         # F.smooth_l1_loss(
#         #     pred_bbox_deltas[sampled_pos_inds],
#         #     regression_targets[sampled_pos_inds],
#         #     beta=1 / 9,
#         #     reduction="sum",
#         # ) / (sampled_inds.numel())

#         objectness_loss = sigmoid_focal_loss(
#             objectness[sampled_inds],
#             labels[sampled_inds],
#             alpha=-1,
#             gamma=2,
#             reduction="sum"
#         ) / (sampled_inds.numel())
#         # originally:
#         # F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

#         return objectness_loss, box_loss

#     def forward(
#         self,
#         images: ImageList,
#         features: dict[str, torch.Tensor],
#         targets: Optional[list[dict[str, torch.Tensor]]] = None,
#     ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:

#         """
#         Copy of torchvision.models.detection.rpn.RegionProposalNetwork.forward
#         adapted for different losses.
#         """
#         features = list(features.values())
#         objectness, pred_bbox_deltas = self.head(features)
#         anchors = self.anchor_generator(images, features)

#         num_images = len(anchors)
#         num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
#         num_anchors_per_level = [
#             s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors
#         ]
#         objectness, pred_bbox_deltas = concat_box_prediction_layers(
#             objectness,
#             pred_bbox_deltas
#         )

#         proposals = self.box_coder.decode(pred_bbox_deltas, anchors)
#         proposals = proposals.view(num_images, -1, 4)
#         boxes, scores = self.filter_proposals(
#             proposals.detach(),
#             objectness.detach(),
#             images.image_sizes,
#             num_anchors_per_level
#         )

#         losses = {}
#         if self.training:
#             if targets is None:
#                 raise ValueError("targets should not be None")
#             labels, matched_gt_boxes = self.assign_targets_to_anchors(
#                 anchors,
#                 targets
#             )


#             loss_objectness, loss_rpn_box_reg = self.compute_loss(
#                 objectness, proposals, labels, matched_gt_boxes
#             )
#             losses = {
#                 "loss_objectness": loss_objectness,
#                 "loss_rpn_box_reg": loss_rpn_box_reg,
#             }
#         return boxes, losses

# tmdf.RegionProposalNetwork = RegionProposalNetwork

# #
# # roi_align using improved alignment from detectron2
# #

# orig_roi_align = top.roi_align

# def detectron_roi_align(*args, **kwargs):
#     if "aligned" not in kwargs:
#         kwargs["aligned"] = True
#     return orig_roi_align(*args, **kwargs)

# top.roi_align = detectron_roi_align

# #
# # alternative iou for rpn.box_similarity to tame OOMs in case of too many ground truths
# #

# @torch.compiler.disable
# def chunked_box_iou(
#         boxes1: torch.Tensor,
#         boxes2: torch.Tensor,
#         chunk_size = 16384 # in isolation 1800 is optimum on 4090, but ultraslow in training
# ) -> torch.Tensor:
#     """
#     Copy of torchvision.ops.boxes.box_iou that chunks the IoU
#     computations. Can be used in case of datasets with a huge number
#     of candidate boxes, which frequently lead to OOMs when processed
#     at once.
#     """
#     if sqrt(boxes1.shape[0]*boxes2.shape[0]) < chunk_size:
#         return box_iou(boxes1, boxes2)

#     if not torch.jit.is_scripting() and not torch.jit.is_tracing():
#         _log_api_usage_once(chunked_box_iou)

#     if boxes2.shape[0] < chunk_size:
#         chunk_y_size = boxes2.shape[0]
#         chunk_x_size = floor(chunk_size / chunk_y_size)
#     elif boxes1.shape[0] < chunk_size:
#         chunk_x_size = boxes1.shape[0]
#         chunk_y_size = floor(chunk_size / chunk_x_size)
#     else:
#         chunk_x_size = chunk_size
#         chunk_y_size = chunk_size

#     iou = torch.zeros(
#         (boxes1.shape[0], boxes2.shape[0]),
#         device=boxes1.device,
#         dtype=boxes1.dtype
#     )

#     i = 0
#     j = 0
#     while i * chunk_x_size < boxes1.shape[0]:
#         x_slice = slice(i*chunk_x_size, (i+1)*chunk_x_size)
#         while j * chunk_y_size < boxes2.shape[0]:
#             y_slice = slice(j*chunk_y_size, (j+1)*chunk_y_size)

#             inter, union = _memopt_box_inter_union(
#                 boxes1[x_slice, :],
#                 boxes2[y_slice, :]
#             )
#             iou[x_slice, y_slice] = inter / union

#             j += 1
#         i += 1

#     return iou


# def _memopt_box_inter_union(
#         boxes1: torch.Tensor,
#         boxes2: torch.Tensor
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     # copy of _box_inter_union that shouldn't create superfluous vars
#     wh = _upcast(
#         torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) -
#         torch.max(boxes1[:, None, :2], boxes2[:, :2])
#     ).clamp(min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

#     union = box_area(boxes1)[:, None] + box_area(boxes2) - inter

#     return inter, union


###
### build helper
###

def build_generalized_frcnn_from_conf(
        class_map: ClassMap,
        backbone: torch.nn.Module,
        anchor_gen: torch.nn.Module,
        rpn_head_partial: Callable[..., torch.nn.Module],
        box_roi_pool: torch.nn.Module,
        box_head_partial: Callable[..., torch.nn.Module],
        mult_obj_loss: float,
        mult_rpn_reg_loss: float,
        mult_cls_loss: float,
        mult_box_reg_loss: float,
        compile_parts: bool,
        **kwargs # args for torchvision.models.detection.faster_rcnn.FasterRCNN
) -> FasterRCNNWrapper:
    rpn_head = rpn_head_partial(
        in_channels=backbone.out_channels,
        num_anchors=anchor_gen.num_anchors_per_location()[0],
    )

    box_head = box_head_partial(
        input_size=(backbone.out_channels, *box_roi_pool.output_size)
    )

    faster_rcnn = tmdf.FasterRCNN(
        backbone=backbone,
        num_classes=class_map.num_classes + 1, # +1 for background
        rpn_anchor_generator=anchor_gen,
        rpn_head=rpn_head,
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        **kwargs
    )

    # # try to get rid of OOMs for images with high number of dets and gts
    # # RegionProposalNetwork.box_similarity seems to be the culprit
    # faster_rcnn.rpn.box_similarity = chunked_box_iou

    return FasterRCNNWrapper(
        model=faster_rcnn,
        class_map=class_map,
        do_input_trafo=True,
        mult_obj_loss=mult_obj_loss,
        mult_rpn_reg_loss=mult_rpn_reg_loss,
        mult_cls_loss=mult_cls_loss,
        mult_box_reg_loss=mult_box_reg_loss,
        compile_parts=compile_parts
    )
