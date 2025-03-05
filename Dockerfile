# To build the container run something like
#    docker build -t name:revision .
# and to run the container execute something like
#    docker run --rm --name container_name --gpus all -p8081:5000 name:revision
#
# Make sure that config/inference.yaml is set up correctly before building,
# e.g. device, weight locations, postprocessing.
#
# To keep the container "lightweight"" take care that requirements.txt
# only holds the requirements needed for inference

FROM ubuntu:22.04

WORKDIR /usr/src/code

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3-pip python3-dev libturbojpeg gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt /usr/src/code
RUN pip3 install -U pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /usr/src/code

EXPOSE 5000

CMD ["gunicorn", "--config=config/gunicorn.py", "run_server:app"]
