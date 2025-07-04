# 🚫 BlurScene — Privacy-preserving Anonymization for Traffic Data

**BlurScene** is a deep learning model designed to **anonymize faces and license plates** in traffic-related images and video data.

---

## 📋 Table of Contents

1. [🔧 Setup](#setup)
    - [📦 Environment](#environment-setup)
2. [🧠 Inference](#inference)
    - [⚙️ Configuration](#configuration)
    - [📥 Model Weights Download](#model-weights-download)
    - [🧪 Inference Script](#inference-script)
    - [🌐 Server](#server)
    - [🐳 Docker](#docker)

---

## 🔧 Setup

### 📦 Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> ✅ Requires Python ≥ 3.10 (tested with 3.10.12)

---

## 🧠 Inference

### ⚙️ Configuration

All inference settings can be found in [`config/inference.yaml`](config/inference.yaml).  
You must provide:

- ✅ Path to the **model weights** `.pt`
- ✅ Path to the **model configuration** `.yaml` (Hydra config)

📁 It's recommended to store both in a `weights/` folder (which is whitelisted in `.dockerignore`).

#### 🧰 Optional Processing Steps:

- `mirror_image`: horizontal mirroring  
- `enlarged_regions_n`: crops the image into _n × n_ subregions

#### 🔎 Post-Processing Parameters:

- `pre_merge_score_threshold`, `post_merge_score_threshold`: filter weak predictions
- `merging_method`:  
  - `wbf`: Weighted Box Fusion  
  - `nms`: Non-Maximum Suppression  
  - `nmm`: Non-Maximum Merging
- `merge_iou_threshold`: IoU threshold for merging
- `area_method`: how box area is computed:
  - `int`: `(x1 - x0 + 1) * (y1 - y0 + 1)`
  - `float`: `(x1 - x0) * (y1 - y0)`

You can also configure logging level and format via the YAML.

---

### 📥 Model Weights Download

You can download the pretrained model weights and matching configuration from the following link:

👉 [**Download BlurScene Weights**](https://cloud.volkmann-sv.de/index.php/s/4m8sfqBPkAdG7rP)

📁 Place the downloaded files inside the `weights/` directory.  
⚠️ Make sure the `.pt` and `.yaml` files match exactly — otherwise inference may fail.

---

### 🧪 Inference Script

You can test inference directly via:

```bash
python inference.py path/to/image.jpg
```

This loads the model, compiles it if necessary, and outputs detections for the image.

---

### 🌐 Server

Start a local Flask server using:

```bash
python run_server.py
```

Or use **Gunicorn** (recommended):

```bash
gunicorn --config=config/gunicorn.py run_server:app
```

🔧 Gunicorn reads environment variables:
- `WORKERS`
- `WORKER_TIMEOUT`
- `PORT`
- `LOG_LEVEL`

Default address is `http://0.0.0.0:5000`.

#### 🧪 Test the Endpoint

```bash
curl -H "Content-Type: image/jpeg" --data-binary @test.jpg http://localhost:5000/anonymize --output returned_image.jpg
```

---

### 🐳 Docker

#### 🏗 Build

```bash
docker build . -t blurscene
```

Note: Settings in `inference.yaml` (e.g. `device`) are **fixed** at build time.

#### 🚀 Run

```bash
docker run -d --gpus all -p 5000:5000 --name blurscene_container blurscene
```

🔍 Monitor logs:

```bash
docker logs -f blurscene_container
```

Wait for model compilation to complete before sending requests.

---

## 📑 Citation

```
@misc{vosshans2024aeifdatacollectiondataset,
    author    = {Marcel Vosshans and Alexander Baumann and Matthias Drueppel and Omar Ait-Aider and Ralf Woerner and Youcef Mezouar and Thao Dang and Markus Enzweiler},
    title     = {The AEIF Data Collection: A Dataset for Infrastructure-Supported Perception Research with Focus on Public Transportation},
    url       = {https://arxiv.org/abs/2407.08261},
    year      = {2024},
}
```

## 📜 License

This project is released under the **[MIT License](LICENSE)**.
