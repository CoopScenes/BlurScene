import cv2
import numpy as np
import torch

from common.type_aliases import (
    GenericBboxesT,
    ImageT,
    LabelsT,
    PredictionT,
    TorchImageT
)


def read_img(image_path: str) -> ImageT:
    """
    Read an image in RGB format, (h, w, 3)
    """
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Unable to open {image_path}")
    if len(bgr_img.shape) == 2 or bgr_img.shape[-1] == 1:
        bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)

    rgb_img = bgr_img[..., (2, 1, 0)]

    return rgb_img


def img_to_torch(img: ImageT) -> TorchImageT:
    """
    Transform a numpy image with rank (h, w, c) to the format usually
    used in torch (c, h, w)
    """
    img_t = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img_t)


def torch_to_img(img: TorchImageT) -> ImageT:
    img = img.numpy()
    return img.transpose((1, 2, 0))


def read_img_torch(image_path: str) -> TorchImageT:
    return img_to_torch(read_img(image_path))


def visualize_bboxes(
        img: ImageT,
        dets: GenericBboxesT,
        color: tuple[int,int,int] = (0, 255, 0)
) -> ImageT:
    img = img.copy()    # don't change original img, and needed for cv2

    for x0, y0, x1, y1, *_ in dets:
        pt0 = (int(x0), int(y0))
        pt1 = (int(x1), int(y1))
        cv2.rectangle(img, pt0, pt1, color, 2)

    return img


def visualize_bboxes_torch(
        img: TorchImageT,
        dets: GenericBboxesT,
        color: tuple[int,int,int] = (0, 255, 0),
) -> ImageT:
    return visualize_bboxes(
        torch_to_img(img),
        dets,
        color
    )


def visualize_gt_pred_torch(
        img: TorchImageT,
        gts: LabelsT,
        preds: PredictionT
) -> ImageT:
    img = visualize_bboxes_torch(img, gts[0].cpu().numpy(), (0, 0, 255))
    img = visualize_bboxes(img, preds[0].cpu().numpy(), (0, 255, 0))
    return img


def text_to_image(
        img: ImageT,
        text: str | list[str],
        color: tuple[int, int, int] = (0,255,0)
) -> ImageT:
    if isinstance(text, list):
        text = "\n".join(text)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 2
    (_, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )
    coords = (10, 10 + baseline + text_height)

    return cv2.putText(
        img,
        text,
        coords,
        font,
        scale,
        color,
        thickness
    )

def write_img_dets(
        img: ImageT,
        det_list: GenericBboxesT,
        outpath: str
) -> None:
    img = visualize_bboxes(img, det_list)
    if not cv2.imwrite(outpath, img):
        raise Exception(f"Unable to write image to {outpath}.")


def write_torch_img_dets(
        img: TorchImageT,
        det_list: GenericBboxesT,
        outpath: str,
) -> None:
    img_np = img.detach().cpu().numpy().astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = img_np[..., (2, 1, 0)]
    write_img_dets(img_np, det_list, outpath)


def write_torch_img_dets_gts(
        img: TorchImageT,
        outpath: str,
        det_list: GenericBboxesT,
        gt_list: GenericBboxesT
) -> None:
    img_np = img.detach().cpu().numpy().astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = img_np[..., (2, 1, 0)]

    img_np = visualize_bboxes(img_np, gt_list, (0, 0, 255))
    img_np = visualize_bboxes(img_np, det_list, (0, 255, 0))

    if not cv2.imwrite(outpath, img_np):
        raise Exception(f"Unable to write image to {outpath}.")
