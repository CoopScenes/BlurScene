#!/usr/bin/env python
"""
Starte den Server z.B. über gunicorn:
    gunicorn --config=config/gunicorn.py run_server:app
"""
import cv2
import numpy as np
import torch
import io
import zipfile

from flask import Flask, request, send_file
from io import BytesIO
from logging import getLogger
from numpy.typing import NDArray

from inference import Inference

logger = getLogger(__name__)
app = Flask(__name__)
anon_ep = "anonymize"

# Inferenz-Instanz initialisieren
inference = Inference()

@app.route("/", methods=["GET"])
def test():
    """Gibt eine einfache HTML-Seite mit Hinweisen zurück."""
    url = request.host_url
    anon_url = f"{url}{anon_ep}"
    msg = f"""
    <!DOCTYPE html>
    <html>
    <body>
    <h1>Anonymization Server</h1>
    <h2>Usage</h2>
    <p>Send an image to {anon_url} for single-image processing or multiple images to /batch_anonymize.</p>
    <h2>Curl Bash Example (Single Image)</h2>
    <p>curl -H "Content-Type: image/jpeg" --data-binary @image.jpg {anon_url} --output returned_image.jpg</p>
    </body>
    </html>
    """
    return msg

@app.route(f"/{anon_ep}", methods=["POST"])
def anon_route():
    """Verarbeitet POST-Anfragen für die Anonymisierung eines einzelnen Bildes."""
    if "image" not in request.content_type:
        return "Unknown content type", 415

    img_size_bytes = request.content_length
    if not img_size_bytes:
        msg = "Image has 0 Bytes."
        logger.error(msg)
        return msg, 400

    if img_size_bytes > 10**8:
        logger.debug(f"Received POST with {request.content_length=}")
        return f"Image too large: {img_size_bytes / 2**20:.2e} MiB", 413

    img_bytes = request.get_data()
    try:
        np_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)[..., (2, 1, 0)]
    except Exception as e:
        logger.exception("Unable to decode the image.")
        return f"Something went wrong: {e}", 500

    bboxes, classes, _ = inference.predict(img)
    face_mask = classes == inference.class_map.name_to_index["face"]
    logger.debug(f"Found {face_mask.sum().item()} faces.")
    logger.debug(f"Found {(~face_mask).sum().item()} license plates.")

    bboxes_np = bboxes.to(dtype=torch.int32).cpu().numpy().astype(np.int32)
    img = anonymize(img, bboxes_np)

    if "jpg" in request.content_type or "jpeg" in request.content_type:
        mime = "image/jpeg"
        suffix = "jpeg"
    else:
        mime = "image/png"
        suffix = "png"

    imenc_ret, img_buf = cv2.imencode(f".{suffix}", img[..., (2,1,0)])
    if not imenc_ret:
        return f"Unable to encode image", 500
    img_buf = BytesIO(img_buf.tobytes())
    return send_file(img_buf, mimetype=mime, download_name=f"anon_image.{suffix}")

@app.route("/batch_anonymize", methods=["POST"])
def batch_anon_route():
    """
    Verarbeitet POST-Anfragen für Batch-Inferenz.
    Erwartet einen multipart/form-data Request mit mehreren Dateien unter dem Schlüssel 'images'.
    """
    files = request.files.getlist("images")
    if not files:
        return "No images uploaded.", 400

    images = []
    for f in files:
        file_bytes = f.read()
        try:
            np_array = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)[..., (2, 1, 0)]
            images.append(img)
        except Exception as e:
            logger.exception("Unable to decode one of the images.")
            return f"Error decoding images: {e}", 500

    # Batch-Inferenz durchführen
    batch_results = inference.predict_batch(images)

    # Erstelle ein ZIP-Archiv mit den anonymisierten Bildern
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, (boxes, _, _) in enumerate(batch_results):
            boxes_np = boxes.to(dtype=torch.int32).cpu().numpy().astype(np.int32)
            anon_img = anonymize(images[idx], boxes_np)
            ret, encoded_img = cv2.imencode(".jpg", anon_img[..., (2,1,0)])
            if ret:
                zf.writestr(f"anon_image_{idx}.jpg", encoded_img.tobytes())
    mem_zip.seek(0)
    return send_file(mem_zip, mimetype="application/zip", as_attachment=True, download_name="anonymized_images.zip")

def anonymize(img: NDArray, dets: NDArray) -> NDArray:
    """
    Wendet einen Mosaic-Style-Effekt auf die durch die Bounding Boxes definierten Bereiche an.
    """
    h, w = img.shape[:2]
    for x0, y0, x1, y1, *_ in dets:
        x_margin = int((x1 - x0) / 10)
        y_margin = int((y1 - y0) / 10)
        x0m = max(x0 - x_margin, 0)
        y0m = max(y0 - y_margin, 0)
        x1m = min(x1 + x_margin, w)
        y1m = min(y1 + y_margin, h)
        anon_box = _anonymize(img[y0m:y1m, x0m:x1m])
        img[y0m:y1m, x0m:x1m] = anon_box
    return img

def _anonymize(crop: NDArray) -> NDArray:
    """
    Wendet eine Mosaik-Anonymisierung auf einen Bildausschnitt an.
    """
    block_size = 5
    h, w = crop.shape[:2]
    im = crop.copy()
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = crop[i:i+block_size, j:j+block_size]
            avg_color = np.mean(block, axis=(0, 1), dtype=int)
            crop[i:i+block_size, j:j+block_size] = avg_color
    mask = _get_elliptical_mask(crop)
    mask = mask[:, :, None]
    crop = (1 - mask) * im + mask * crop
    return np.round(crop).astype(int)

def _get_elliptical_mask(img: NDArray) -> NDArray:
    kx = int(img.shape[1] / 20)
    ky = int(img.shape[0] / 20)
    kx = kx if kx % 2 == 1 else kx + 1
    ky = ky if ky % 2 == 1 else ky + 1
    kx = min(kx, 11)
    ky = min(ky, 11)
    m = np.zeros(img.shape[:2])
    center = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    axes = (img.shape[1] - kx, img.shape[0] - ky)
    r = (center, axes, 0)
    m = cv2.ellipse(m, r, 1, -1)
    m = cv2.blur(m, (kx, ky), borderType=cv2.BORDER_CONSTANT)
    return m

if __name__ == "__main__":
    app.run()
