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
import time

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
    <p>Send an image to {anon_url} for single-image processing or multiple images to /batch_anonymize.</p>
    <h2>Curl Bash Example (Single Image)</h2>
    <p>curl -H "Content-Type: image/png" --data-binary @image.png {anon_url} --output returned_image.png</p>
    </body>
    </html>
    """
    return msg


@app.route(f"/{anon_ep}", methods=["POST"])
def anon_route():
    """Handle POST requests for anonymizing images."""
    if "image" not in request.content_type:
        return "Unknown content type", 415

    # Check upload size
    img_size_bytes = request.content_length
    if not img_size_bytes:
        msg = "Image has 0 Bytes."
        logger.error(msg)
        return msg, 400

    if img_size_bytes > 10 ** 8:
        logger.debug(f"Received POST with {request.content_length=}")
        return f"Image too large: {img_size_bytes / 2 ** 20:.2e} MiB", 413

    img_bytes = request.get_data()

    # image to numpy array
    try:
        np_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)[..., (2, 1, 0)]
    except Exception as e:
        logger.exception("Unable to decode the image.")
        return f"Something went wrong: {e}", 500

    # Modell-Inferenz durchführen
    bboxes, classes, _ = inference.predict(img)
    # Debug-Ausgaben
    face_mask = classes == inference.class_map.name_to_index["face"]
    logger.debug(f"Found {face_mask.sum().item()} faces.")
    logger.debug(f"Found {(~face_mask).sum().item()} license plates.")

    # Konvertierung der Vorhersagen in numpy-Arrays
    bboxes_np = bboxes.to(dtype=torch.int32).cpu().numpy().astype(np.int32)
    classes_np = classes.cpu().numpy().astype(np.int32)

    # Verwende die erweiterte anonymize-Funktion, die auch die Klassen berücksichtigt
    img = anonymize(img, bboxes_np, classes_np)

    # Auswahl des passenden MIME-Typs
    if "jpg" in request.content_type or "jpeg" in request.content_type:
        mime = "image/jpeg"
        suffix = "jpeg"
    else:
        mime = "image/png"
        suffix = "png"

    imenc_ret, img_buf = cv2.imencode(f".{suffix}", img[..., (2, 1, 0)])
    if not imenc_ret:
        return "Unable to encode image", 500
    img_buf = BytesIO(img_buf.tobytes())
    return send_file(img_buf, mimetype=mime, download_name=f"anon_image.{suffix}")


@app.route("/batch_anonymize", methods=["POST"])
def batch_anon_route():
    total_start = time.perf_counter()

    files = request.files.getlist("images")
    if not files:
        return "No images uploaded.", 400

    images = []
    filenames = []
    for f in files:
        file_bytes = f.read()
        try:
            np_array = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)[..., (2, 1, 0)]
            images.append(img)
            filenames.append(f.filename)
        except Exception as e:
            app.logger.exception("Unable to decode one of the images.")
            return f"Error decoding images: {e}", 500

    # Inferenz im Batch
    model_start = time.perf_counter()
    batch_results = inference.predict_batch(images)
    model_end = time.perf_counter()
    model_duration = model_end - model_start

    # Postprocessing: ZIP-Archiv erstellen
    post_start = time.perf_counter()
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for i, (boxes, classes, _) in enumerate(batch_results):
            boxes_np = boxes.to(dtype=torch.int32).cpu().numpy().astype(np.int32)
            classes_np = classes.cpu().numpy().astype(np.int32)
            anon_img = anonymize(images[i], boxes_np, classes_np)
            ret, encoded_img = cv2.imencode(".png", anon_img[..., (2, 1, 0)])
            if ret:
                base_name = filenames[i].rsplit('.', 1)[0]
                zf.writestr(f"{base_name}.PNG", encoded_img.tobytes())
    post_end = time.perf_counter()
    post_duration = post_end - post_start

    total_end = time.perf_counter()
    total_duration = total_end - total_start

    app.logger.info(
        f"Batch Processing Time: Inference: {model_duration:.3f}s, Postprocessing: {post_duration:.3f}s, Total: {total_duration:.3f}s")

    mem_zip.seek(0)
    return send_file(mem_zip, mimetype="application/zip", as_attachment=True, download_name="anonymized_images.zip")


def anonymize(img: NDArray, dets: NDArray, classes: NDArray) -> NDArray:
    """
    Wendet einen Mosaik- (Pixelations-) Effekt inklusive weichem Übergang (Blending) auf
    Bildregionen an, die durch Bounding Boxes definiert sind.
    Dabei werden unterschiedliche Maskentypen (ellipse / rectangle) basierend auf der Objektklasse verwendet.

    Args:
        img: Eingabebild als ndarray (H, W, 3), dtype=uint8.
        dets: Array von Bounding Boxes im Format [x0, y0, x1, y1, ...].
        classes: Array von Klassennummern zu den detektierten Objekten.

    Returns:
        Das anonymisierte Bild.
    """
    h, w = img.shape[:2]

    for idx, det in enumerate(dets):
        x0, y0, x1, y1, *_ = det

        # Berechne die Breite und Höhe der Bounding Box
        bbox_width = x1 - x0
        bbox_height = y1 - y0

        # Wähle Divisor und minimale Blockgröße anhand der Objektklasse:
        if classes[idx] == inference.class_map.name_to_index["face"]:
            divisor = 8    # Beispiel: Bei Gesichtern
            min_block = 5  # Minimalwert für Gesichter
        else:
            divisor = 15   # Beispiel: Bei Kennzeichen
            min_block = 3  # Minimalwert für Kennzeichen

        # Berechnung der Blockgröße unter Verwendung des ausgewählten Divisors
        block_size = int(min(max(bbox_width, bbox_height) / divisor, 30))
        block_size = max(block_size, min_block)

        # Vergrößern der Bounding Box um einen kleinen Rand
        x_margin = int(bbox_width / 10)
        y_margin = int(bbox_height / 10)
        x0m = max(x0 - x_margin, 0)
        y0m = max(y0 - y_margin, 0)
        x1m = min(x1 + x_margin, w)
        y1m = min(y1 + y_margin, h)

        if y1m - y0m < 2 or x1m - x0m < 2:
            continue

        # Auswahl des Maskentyps anhand der Klasse
        if classes[idx] == inference.class_map.name_to_index["face"]:
            mask_type = "ellipse"
        else:
            mask_type = "rectangle"

        anon_box = _anonymize_with_mask(img[y0m:y1m, x0m:x1m], block_size, mask_type)
        img[y0m:y1m, x0m:x1m] = anon_box

    return img

def _anonymize_with_mask(crop: NDArray, block_size: int, mask_type: str) -> NDArray:
    """
    Wendet den Mosaikeffekt auf einen Bildausschnitt an und blended
    diesen je nach mask_type entweder mit einer elliptischen oder rechteckigen
    Maske.

    Args:
        crop: Bildausschnitt, der anonymisiert werden soll.
        block_size: Größe der Blöcke für die Pixelation.
        mask_type: "ellipse" oder "rectangle", je nach gewünschtem Übergang.

    Returns:
        Der anonymisierte Ausschnitt.
    """
    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return crop  # Fehlervermeidung bei zu kleinen Regionen

    # Originalkopie des Ausschnitts
    original_crop = crop.copy()
    # Mosaik-Pixelation anwenden
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = crop[i:i + block_size, j:j + block_size]
            avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
            crop[i:i + block_size, j:j + block_size] = avg_color

    # Auswahl der Maske
    if mask_type == "ellipse":
        mask = _get_elliptical_mask(crop)
    elif mask_type == "rectangle":
        mask = _get_rectangular_mask(crop)
    else:
        mask = np.ones((h, w), dtype=np.float32)
    mask = mask[:, :, None].astype(np.float32)

    # Weiches Mischen: Original und pixelierter Ausschnitt werden anhand der Maske kombiniert
    blended = original_crop.astype(np.float32) * (1 - mask) + crop.astype(np.float32) * mask
    return np.round(np.clip(blended, 0, 255)).astype(np.uint8)


def _get_elliptical_mask(img: NDArray) -> NDArray:
    """
    Erzeugt eine weiche elliptische Maske im Bereich [0, 1] zur Blendung.

    Args:
        img: Der Bildausschnitt, für den die Maske erzeugt wird.

    Returns:
        2D Maske als float32-Array.
    """
    h, w = img.shape[:2]
    kx = int(w / 20)
    ky = int(h / 20)
    kx = kx if kx % 2 == 1 else kx + 1
    ky = ky if ky % 2 == 1 else ky + 1
    kx = min(kx, 11)
    ky = min(ky, 11)

    m = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = ((w - kx) // 2, (h - ky) // 2)

    # Fallback für sehr kleine Ausschnitte
    if axes[0] <= 0 or axes[1] <= 0:
        axes = (w // 2, h // 2)

    cv2.ellipse(m, center, axes, 0, 0, 360, 1, -1)
    m = cv2.blur(m.astype(np.float32), (kx, ky), borderType=cv2.BORDER_CONSTANT)
    return m


def _get_rectangular_mask(img: NDArray) -> NDArray:
    """
    Erzeugt eine rechteckige Maske mit weichen Übergängen im Bereich [0, 1].
    Dabei wird zunächst im Zentrum ein voll anonymisierter (Wert 1) Bereich gesetzt,
    während an den Rändern der Wert 0 liegt. Anschließend wird der Übergang mit Blur geglättet.

    Args:
        img: Der Bildausschnitt, für den die Maske erzeugt wird.

    Returns:
        Eine 2D-Maske als float32-Array.
    """
    h, w = img.shape[:2]
    # Definiere die Breite des Randes (z. B. 10 % der kleineren Dimension, mindestens 1 Pixel)
    border_thickness = max(int(min(w, h) / 10), 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Setze den zentralen Bereich auf 1
    mask[border_thickness:h - border_thickness, border_thickness:w - border_thickness] = 1
    # Glätte den Randbereich mit Blur
    kernel_size = (border_thickness * 2 + 1, border_thickness * 2 + 1)
    mask = cv2.blur(mask.astype(np.float32), kernel_size, borderType=cv2.BORDER_CONSTANT)
    mask = np.clip(mask, 0, 1)
    return mask


if __name__ == "__main__":
    app.run()