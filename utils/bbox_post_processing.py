import argparse
import torch

from collections.abc import Sequence
from torchvision.ops import nms, box_iou as iou_float

from common.type_aliases import BatchPredictionT
from utils.misc import load_predictions_hdf, save_predictions_hdf


class Postprocessor():
    def __init__(
            self,
            merge_iou_thresh: float = 0.5,       # at which iou to merge bboxes
            pre_merge_score_thresh: float = 0,   # applied before merging algo
            post_merge_score_thresh: float = 0,  # applied to merged bboxes
            merging_method: str = "wbf",
            area_method: str = "int"             # if areas are computed with integer or float coordinates
    ):
        if merging_method not in ["nms", "nmm", "wbf"]:
            raise ValueError(f"Unknown merging method: {merging_method=}")
        self.merging_fn = getattr(self, merging_method)

        self.iou_thresh = merge_iou_thresh
        self.pre_thresh = pre_merge_score_thresh
        self.post_thresh = post_merge_score_thresh

        self.area_method = area_method
        if area_method == "float":
            self.iou_fn = iou_float
            self.area_fn = type(self).area_float
        elif area_method == "int":
            self.iou_fn = type(self).iou_int
            self.area_fn = type(self).area_int
        else:
            raise ValueError(f"Unknown area method: {area_method=}")

    @staticmethod
    def bboxes_to_int(bboxes: torch.Tensor) -> torch.Tensor:
        bboxes[:, 0] = torch.floor(bboxes[:, 0])
        bboxes[:, 1] = torch.floor(bboxes[:, 1])
        bboxes[:, 2] = torch.ceil(bboxes[:, 2])
        bboxes[:, 3] = torch.ceil(bboxes[:, 3])
        bboxes = bboxes.to(torch.int64)
        return bboxes

    @classmethod
    def iou_int(
            cls,
            bboxes0: torch.Tensor,
            bboxes1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU of bboxes0 and bboxes1 using integer coordinates.

        This implementation assigns an area of 1 to boxes where the left upper
        corner coordinates are equal to the right lower corner coordinates.
        """
        bboxes0 = cls.bboxes_to_int(bboxes0)
        bboxes1 = cls.bboxes_to_int(bboxes1)

        x0 = torch.maximum(bboxes0[:, 0][:, None], bboxes1[:, 0])
        x1 = torch.minimum(bboxes0[:, 2][:, None], bboxes1[:, 2])
        y0 = torch.maximum(bboxes0[:, 1][:, None], bboxes1[:, 1])
        y1 = torch.minimum(bboxes0[:, 3][:, None], bboxes1[:, 3])

        inter_width = torch.maximum(torch.zeros_like(x1), x1 - x0 + 1)
        inter_height = torch.maximum(torch.zeros_like(y1), y1 - y0 + 1)
        interarea = inter_width * inter_height

        area0 = cls.area_int(bboxes0)
        area1 = cls.area_int(bboxes1)
        unionarea = area0[:, None] + area1 - interarea

        return torch.where(
            unionarea > 0,
            interarea / unionarea,
            torch.zeros_like(interarea)
        )

    @staticmethod
    def area_int(bboxes: torch.Tensor):
        return (
            (bboxes[:, 2] - bboxes[:, 0] + 1) *
            (bboxes[:, 3] - bboxes[:, 1] + 1)
        )

    @staticmethod
    def area_float(bboxes: torch.Tensor):
        return (
            (bboxes[:, 2] - bboxes[:, 0]) *
            (bboxes[:, 3] - bboxes[:, 1])
        )

    def nmm(
            self,
            bboxes: torch.Tensor,
            scores: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """
        Non-Maximum Merging of all of the bboxes in the input to a single bbox.
        The resulting bbox is the enveloping box of all boxes. The new score is
        a sum of all scores weighted by the boxes' areas.
        """
        if bboxes.shape[0] == 1:
            return bboxes[0], scores[0]

        areas = self.area_fn(bboxes)
        total_area = areas.sum()

        new_x0 = bboxes[:, 0].min()
        new_y0 = bboxes[:, 1].min()
        new_x1 = bboxes[:, 2].max()
        new_y1 = bboxes[:, 3].max()
        new_score = (scores * areas).sum() / total_area

        return (
            torch.tensor(
                [new_x0, new_y0, new_x1, new_y1],
                dtype=bboxes.dtype,
                device=bboxes.device
            ),
            new_score
        )

    def wbf(
            self,
            bboxes: torch.Tensor,
            scores: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """
        Weighted Box Fusion averages over all boxes.
        """
        if bboxes.shape[0] == 1:
            return bboxes[0], scores[0]

        new_bbox = torch.sum(bboxes * scores[:, None], dim=0) / scores.sum()

    #    new_score = scores.mean() # in the paper new_score is a simple mean...
        new_score = (scores**2).sum() / scores.sum() # ...and i think that's wrong

        return new_bbox, new_score

    def nms(
            self,
            bboxes: torch.Tensor,
            scores: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Non-Maximum Suppression keeps only the highest scoring box.
        """
        if self.area_method != "float":
            raise NotImplementedError("NMS is implemented only for float.")
        keep_idcs = nms(bboxes, scores, self.iou_thresh)
        return bboxes[keep_idcs[0]], scores[keep_idcs[0]]

    @staticmethod
    def _flatten_predictions(
            preds_batch: BatchPredictionT,
    ):
        """
        Take predictions for a batch in form of a list of tuples of tensors
            preds_batch=[(bboxes0, classes0, scores0), (bboxes1, ...), ...]
        and return a concatenated tensor for each of bboxes, classes, scores and
        a category index to that encodes class and batch index of each bounding box.
        """
        batch_size = len(preds_batch)

        bboxes_list = [preds[0] for preds in preds_batch]
        classes_list = [preds[1] for preds in preds_batch]
        scores_list = [preds[2] for preds in preds_batch]

        # generate batch index for each bbox
        lengths = [len(bbs) for bbs in bboxes_list]
        cat_idx = torch.empty(
            (sum(lengths),),
            dtype=torch.int64,
            device=bboxes_list[0].device
        )
        i = 0
        for b, l in enumerate(lengths):
            cat_idx[i:i+l] = b
            i += l

        bboxes = torch.cat(bboxes_list)
        classes = torch.cat(classes_list)
        scores = torch.cat(scores_list)

        # 2d (batch and class) to 1d index; cat_idx = batch_idx until this point
        cat_idx = classes * batch_size + cat_idx

        return cat_idx, bboxes, classes, scores

    @staticmethod
    def _unflatten_categories(
            batch_size: int,
            results_list: Sequence[tuple[torch.Tensor, int, torch.Tensor]], # list of (bbs, cls, scs)
            cats: torch.Tensor # category index of each entry in results_list
    ) -> BatchPredictionT:
        """
        Restore batch dimension as a list from a list of (bboxes, classes, ...)
        and the corresponding category.
        """
        b_batch = [[] for _ in range(batch_size)]
        c_batch = [[] for _ in range(batch_size)]
        s_batch = [[] for _ in range(batch_size)]

        for cat_idx, (bbs, cls, scs) in zip(cats, results_list):
            batch_idx = cat_idx - cls * batch_size
            cls_tensor = torch.ones(
                bbs.shape[0],
                dtype=torch.int64,
                device=bbs.device
            ) * cls
            b_batch[batch_idx].append(bbs)
            c_batch[batch_idx].append(cls_tensor)
            s_batch[batch_idx].append(scs)

        preds = []
        for bbs, cls, scs in zip(b_batch, c_batch, s_batch):
            if len(bbs) > 0:
                bbs_cat = torch.cat(bbs)
                cls_cat = torch.cat(cls)
                scs_cat = torch.cat(scs)
            else:
                bbs_cat = torch.empty((0,4), device=cats.device)
                cls_cat = torch.empty((0,), dtype=torch.int64, device=cats.device)
                scs_cat = torch.empty((0,), device=cats.device)

            preds.append(
                (
                    bbs_cat,
                    cls_cat,
                    scs_cat,
                )
            )

        return preds

    def _merge_bboxes_category(
            self,
            bboxes: torch.Tensor,
            scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sorts all bboxes by score, computes IoU of each box with other candidate
        boxes and passes all candidates that will be merged into a single box on
        to the merging algorithm.
        """
        if bboxes.shape[0] == 0:
            return (bboxes, scores)

        sorted_idxs = torch.argsort(scores, descending=True)

        new_bboxes = []
        new_scores = []

        ious = self.iou_fn(bboxes, bboxes)
        merged_mask = torch.zeros(
            bboxes.shape[0],
            device=bboxes.device,
            dtype=bool
        )

        for main_idx in sorted_idxs:
            if merged_mask[main_idx]:
                continue

            to_merge_mask = (ious[main_idx] >= self.iou_thresh) & ~merged_mask

            bbox, score = self.merging_fn(
                bboxes[to_merge_mask],
                scores[to_merge_mask]
            )

            # update merged mask
            merged_mask = merged_mask | to_merge_mask

            if score > self.post_thresh:
                new_bboxes.append(bbox[None, ...])
                new_scores.append(score[None])

        if len(new_bboxes) == 0:
            return (
                torch.zeros((0,4), dtype=bboxes.dtype, device=bboxes.device),
                torch.zeros((0,), dtype=scores.dtype, device=scores.device)
            )

        return (
            torch.cat(new_bboxes, dim=0),
            torch.cat(new_scores, dim=0)
        )

    @torch.autograd.grad_mode.inference_mode
    def post_processing(
            self,
            preds_batch: BatchPredictionT,
    ) -> BatchPredictionT:
        cats, bbs, cls, scs = self._flatten_predictions(preds_batch)

        # prune by score
        scs[scs < self.pre_thresh] = 0

        cats = cats[scs > 0]
        bbs = bbs[scs > 0]
        cls = cls[scs > 0]
        scs = scs[scs > 0]

        if cats.shape[0] == 0:
            return [
                (
                    torch.empty((0,4), device=cats.device),
                    torch.empty((0,), dtype=torch.int64, device=cats.device),
                    torch.empty((0,), device=cats.device)
                )
                for _ in range(len(preds_batch))
            ]

        # run merging algo on each category
        merged = []
        cats_unique = cats.unique()
        for cidx in cats_unique:
            mask = cats == cidx

            bbs_merged, scs_merged = self._merge_bboxes_category(
                bbs[mask],
                scs[mask]
            )

            merged.append((bbs_merged, cls[mask][0], scs_merged))

        return self._unflatten_categories(
            len(preds_batch),
            merged,
            cats_unique
        )


iou_int = Postprocessor.iou_int

# def get_random_batch(batch_size, max_bbs):
#     batch = []
#     for _ in range(batch_size):
#         n = torch.randint(max_bbs+1, (1,))[0]

#         bboxes = torch.rand((n, 4))
#         bboxes[:, 2:] += 1.
#         bboxes[:, 0] *= 1920 / 2
#         bboxes[:, 2] *= 1920 / 2
#         bboxes[:, 1] *= 1200 / 2
#         bboxes[:, 3] *= 1200 / 2

#         classes = torch.randint(2, (n,))

#         scores = torch.rand((n,))

#         shifted_bboxes = bboxes + 100
#         shifted_bboxes[:, (0,2)] = torch.clamp(shifted_bboxes[:, (0,2)], max=1920)
#         shifted_bboxes[:, (1,3)] = torch.clamp(shifted_bboxes[:, (1,3)], max=1200)

#         batch.append(
#             (
#                 torch.cat([bboxes, shifted_bboxes]),
#                 torch.cat([classes,classes]),
#                 torch.cat([scores,scores / 2])
#             )
#         )

#     return batch


# def test():
#     preds_batch = get_random_batch(8, 4)
#     print([x[1].dtype for x in preds_batch])
#     imgs = torch.zeros((8, 3, 1200, 1920))
#     # preds_batch_merged = prediction_post_processing(preds_batch)
#     postproc = Postprocessor()
#     preds_batch_merged = postproc.post_processing(preds_batch)
#     print([x[1].dtype for x in preds_batch_merged])
#     breakpoint()
#     for i, (img, bbs, bbs_merged) in enumerate(zip(imgs, preds_batch, preds_batch_merged)):
#         write_torch_img_dets(img, bbs[0], f"bb_img_{i}_premerge.jpg")
#         write_torch_img_dets(img, bbs_merged[0], f"bb_img_{i}_merged.jpg")


def prediction_postprocessing(
        hdf_infile: str,
        hdf_outfile: str,
        merge_iou_thresh: float,
        pre_merge_score_thresh: float,
        post_merge_score_thresh: float,
        merging_method: str,
        area_method: str
):
    batch_size = 32

    proc = Postprocessor(
        merge_iou_thresh=merge_iou_thresh,
        pre_merge_score_thresh=pre_merge_score_thresh,
        post_merge_score_thresh=post_merge_score_thresh,
        merging_method=merging_method,
        area_method=area_method,
    )

    d = load_predictions_hdf(hdf_infile)

    batch = []
    new_bbs = []
    new_cls = []
    new_scs = []
    for bbs, cls, scs in zip(d["bboxes"], d["classes"], d["scores"]):
        batch.append((torch.tensor(bbs), torch.tensor(cls), torch.tensor(scs)))

        if len(batch) == batch_size:
            new_batch = proc.post_processing(batch)

            new_bbs.extend([x[0].numpy() for x in new_batch])
            new_cls.extend([x[1].numpy() for x in new_batch])
            new_scs.extend([x[2].numpy() for x in new_batch])
            batch = []

    if len(batch) > 0:
        new_batch = proc.post_processing(batch)

        new_bbs.extend([x[0].numpy() for x in new_batch])
        new_cls.extend([x[1].numpy() for x in new_batch])
        new_scs.extend([x[2].numpy() for x in new_batch])

    if len(new_bbs) != len(d["bboxes"]):
        raise RuntimeError("Inconsistent number of merged predictions.")

    save_predictions_hdf(hdf_outfile, d["data_ids"], new_bbs, new_cls, new_scs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outfile")
    parser.add_argument("merge_iou_thresh", type=float)
    parser.add_argument("pre_merge_score_thresh", type=float)
    parser.add_argument("post_merge_score_thresh", type=float)
    parser.add_argument("merging_method")
    parser.add_argument("area_method")
    args = parser.parse_args()

    prediction_postprocessing(
        args.infile,
        args.outfile,
        args.merge_iou_thresh,
        args.pre_merge_score_thresh,
        args.post_merge_score_thresh,
        args.merging_method,
        args.area_method
    )
