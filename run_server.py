#!/usr/bin/env python

"""
Run via
    flask --app {__file__wo_suffix} run
"""

import cv2
import numpy as np
import torch

from flask import Flask, request, send_file
from io import BytesIO
from logging import getLogger
from numpy.typing import NDArray

from inference import Inference


logger = getLogger(__name__)

app = Flask(__name__)
anon_ep = "anonymize"

inference = Inference()


@app.route("/", methods=["GET"])
def test():
    """Return a basic HTML page with usage instructions."""
    url = request.host_url
    anon_url = f"{url}{anon_ep}"
    msg = f"""
    <!DOCTYPE html>
    <html>
    <body>

    <h1>Anonymization Server</h1>
    <h2>Usage</h2>
    <p>Send an image to {anon_url}.</p>
    <h2>Curl Bash Example</h2>
    <p>curl -H "Content-Type: image/jpeg" --data-binary @image.jpg {anon_url} --output returned_image.jpg</p>
    </body>
    </html>
    """
    return msg


@app.route(f"/{anon_ep}", methods=["POST"])
def anon_route():
    """Handle POST requests for anonymizing images."""
    if "image" not in request.content_type:
        return "Unknown content type", 415

    # check upload size
    img_size_bytes = request.content_length
    if not img_size_bytes:
        msg = "Image has 0 Bytes."
        logger.error(msg)
        return msg, 400

    if img_size_bytes > 10**8:
        logger.debug(f"Received POST with {request.content_length=}")
        return f"Image too large with {img_size_bytes / 2**20:.2e}MiB", 413
    img_bytes = request.get_data()

    # image to numpy array
    try:
        np_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)[..., (2, 1, 0)]
    except Exception as e:
        logger.exception("Unable to decode the image.")
        return f"Something went wrong: {e}", 500

    # get dets
    bboxes, classes, _ = inference.predict(img)

    face_mask = classes == inference.class_map.name_to_index["face"]
    logger.debug(f"Found {face_mask.sum().item()} faces.")
    logger.debug(f"Found {(~face_mask).sum().item()} license plates.")

    # to numpy for blurring
    bboxes = bboxes.to(dtype=torch.int32).cpu().numpy().astype(np.int32)

    # blur and mark dets
    img = anonymize(img, bboxes)

    if "jpg" in request.content_type or "jpeg" in request.content_type:
        mime = "image/jpeg"
        suffix = "jpeg"
    else:
        mime = "image/png"
        suffix = "png"

    imenc_ret, img_buf = cv2.imencode(f".{suffix}", img[..., (2,1,0)])
    if not imenc_ret:
        return (
            f"Unable to encode image, {type(img)=}, {img.dtype=}, {img.shape=}",
            500
        )
    img_buf = BytesIO(img_buf.tobytes())

    return send_file(img_buf, mime, download_name=f"anon_image.{suffix}")


def anonymize(img: NDArray, dets: NDArray) -> NDArray:
    """
    Anonymize regions in the image based on bounding boxes.

    WARNING:
    Modifies img.
    Args:
        img: Input image to be anonymized.
        dets: Array of bounding boxes to anonymize.

    Returns:
        Image with anonymized regions.
    """
    # TODO vectorize if seriously used
    h, w = img.shape[:2]
    for x0, y0, x1, y1, *_ in dets:
        x_margin = int((x1 - x0) / 10)
        y_margin = int((y1 - y0) / 10)
        x0m = x0 - x_margin
        x1m = x1 + x_margin
        y0m = y0 - y_margin
        y1m = y1 + y_margin
        x0m = x0m if x0m > 0 else 0
        x1m = x1m if x1m < w else w
        y0m = y0m if y0m > 0 else 0
        y1m = y1m if y1m < h else h
        anon_box = _anonymize(img[y0m:y1m, x0m:x1m])
        img[y0m:y1m, x0m:x1m] = anon_box

    return img


def _anonymize(crop: NDArray) -> NDArray:
    """
    Apply mosaic-style anonymization to an image crop.

    Args:
        crop: Region of the image to be obfuscated.

    Returns:
        Anonymized crop.
    """
    block_size = 5

    h, w = crop.shape[:2]
    im = crop.copy() # keep original region for mask overlay

    # Apply mosaic effect block-wise
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = crop[i:i+block_size, j:j+block_size]
            avg_color = np.mean(block, axis=(0, 1), dtype=int)
            crop[i:i+block_size, j:j+block_size] = avg_color

    # overlay the mosaic over the original crop, such that the
    # transition towards the edges is smooth
    mask = _get_elliptical_mask(crop)
    mask = mask[:, :, None]
    crop = (1 - mask) * im + mask * crop
    crop = np.round(crop).astype(int)

    return crop


def _get_linear_mask(img: NDArray) -> NDArray:
    """
    Generate a mask with values in [0,1] for smooth transitions
    on the edges of img.

    Args:
        img: Input image for which the mask is generated.
    """
    mask = np.ones(img.shape[:2])
    x_margin = int(img.shape[1] / 10)
    y_margin = int(img.shape[0] / 10)
    x_margin = x_margin if x_margin < 25 else 25
    y_margin = y_margin if y_margin < 25 else 25

    if x_margin > 3:
        x_border = np.linspace(0, 1, x_margin)
        mask[:, :x_margin] = x_border
        mask[:, -x_margin:] = x_border[::-1]

    if y_margin > 3:
        y_border = np.linspace(0, 1, y_margin)
        mask[:y_margin] = mask[:y_margin] * y_border[:, None]
        mask[-y_margin:] = mask[-y_margin:] * y_border[::-1, None]

    return mask


def _get_elliptical_mask(img: NDArray) -> NDArray:
    """
    Generate an elliptical mask with values in [0,1] for smooth
    transitions along the shape of an ellipse.

    Args:
        img: Input image for which the mask is generated.

    Returns:
        Elliptical mask with values in [0,1].
    """
    kx = int(img.shape[1]/20)
    ky = int(img.shape[0]/20)
    kx = kx if kx % 2 == 1 else kx+1
    ky = ky if ky % 2 == 1 else ky+1
    kx = kx if kx < 11 else 11
    ky = ky if ky < 11 else 11

    m = np.zeros(img.shape[:2])
    center = (int(img.shape[1]/2), int(img.shape[0]/2))
    axes = (img.shape[1] - kx, img.shape[0] - ky)
    r = (center, axes, 0)
    m = cv2.ellipse(m, r, 1, -1)

    m = cv2.blur(m, (kx, ky), borderType=cv2.BORDER_CONSTANT)

    return m


if __name__ == "__main__":
    app.run()
