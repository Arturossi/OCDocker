FROM mambaorg/micromamba:1.5.8

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openbabel \
        libopenbabel-dev \
        swig \
        cmake \
        g++ \
        dssp \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n ocdocker -f /tmp/environment.yml \
    && micromamba clean -a -y

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

COPY . /workspace
RUN micromamba run -n ocdocker pip install -e .

CMD ["bash"]
