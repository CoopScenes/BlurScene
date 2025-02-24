# BlurScene
=======

# Table of Contents

1.  [Setup](#org2c8b1f2)
    1.  [Environment Setup](#org5d942a7)
2.  [Inference](#org179d296)
    1.  [Configuration](#org9056a19)
    2.  [Inference Script](#orgf056ac6)
    3.  [Server](#org201a534)
    4.  [Docker Container](#orgb486795)

BlurScene is a model to anonymize faces and license plates of traffic data.


<a id="org2c8b1f2"></a>

# Setup


<a id="org5d942a7"></a>

## Environment Setup

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

should suffice given a Python version >= 3.10.
The code was written using Python 3.10.12.


<a id="org179d296"></a>

# Inference


<a id="org9056a19"></a>

## Configuration

The configuration for inference, i.e. for the files [inference.py](inference.py) and [run\_server.py](run_server.py), can be found in [config/inference.yaml](config/inference.yaml).

The paths to the model weights (.pt) and model configuration (.yaml, a hydra config) have to be given separately. Since the model configuration file completely defines the model architecture, one has to be sure that the conf fits the weights. Additionally, [.dockerignore](.dockerignore) whitelists a `weights` directory, so it is easiest to put model weights and config there. The model device can be specified separately in [config/inference.yaml](config/inference.yaml).

Additional processing steps can be configured in the `processing` section. It is possible to send the image in multiple variants through the model and gather the detections from each variant. There a two options:

-   `mirror_image` which mirrors the image horizontally,
-   `enlarged_regions_n` which slices the image into $n^2$ parts (i.e. the `_n` defines the slices per dimension).

Furthermore, there are the usual post-processing options, i.e.

-   `pre_merge_score_threshold`, `post_merge_score_threshold` throw away predictions below the given scores
-   `merging_method` sets a merging algorithm, one of
    -   Weighted Box Fusion, which computes an average box from overlapping boxes,
    -   Non-Maximum Suppression, which only keeps the highest scoring of overlapping boxes, and
    -   Non-Maximum Merging, which produces the enveloping box.

The `merge_iou_threshold` defines at which overlap (Intersection over Union, IoU) boxes must be merged. `area_method` changes how the box area for the IoU is computed, either

-   `int` is $A = (x_1 - x_0 + 1) * (y_1 - y_0 + 1)$
-   `float` is $A = (x_1 - x_0) * (y_1 - y_0)$.

At last one can define the logging level (the usual logging-module levels) and the log format.


<a id="orgf056ac6"></a>

## Inference Script

[inference.py](inference.py) is a module used by [run\_server.py](run_server.py). However, for testing purposes it is executable. Simply running `python inference.py path/to/image.jpg` should load the model, warm up/compile it, and predict bounding boxes for the given image.


<a id="org201a534"></a>

## Server

The [run\_server.py](run_server.py) script starts a local Flask server for testing. For proper worker handling, etc. the server should be started using gunicorn, `gunicorn --config=config/gunicorn.py run_server:app`. gunicorn can be configured in [config/gunicorn.py](config/gunicorn.py) (see the [gunicorn documentation](https://docs.gunicorn.org/en/stable/settings.html) for details). Here it is pre-configured to query the environment variables  `WORKERS`, `WORKER_TIMEOUT`, `PORT` and `LOG_LEVEL`. By default the gunicorn server has the address `0.0.0.0:5000`, i.e. it opens the port 5000 to the network side. Flask on the other hand runs with its default configuration and opens port 5000 only for connections from localhost.

The anonymization endpoint is `/anonymize` and setup can be tested with e.g. `curl -H "Content-Type: image/jpeg" --data-binary @test.jpg http://localhost:5000/anonymize --output returned_image.jpg`.


<a id="orgb486795"></a>

## Docker Container

The docker container is built by `docker build . -t $image_name` and can be run with e.g. `docker run -d --gpus all -p 5000:5000 --name $container_name $image_name`. With `docker logs -f $container_name` one can monitor the logs and see when the model is ready.

