FROM mambaorg/micromamba:1.5.8

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV PYTHONUNBUFFERED=1
ENV OCDOCKER_CONFIG=/etc/ocdocker/OCDocker.cfg
ENV PATH=/opt/ocdocker/bin:/usr/local/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        cmake \
        dssp \
        graphviz \
        libxrender1 \
        libxext6 \
        openbabel \
        postgresql-client \
        default-mysql-client \
        tar \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n ocdocker -f /tmp/environment.yml \
    && micromamba clean -a -y

WORKDIR /opt/ocdocker
COPY . /opt/ocdocker
RUN micromamba run -n ocdocker pip install -e ".[all]"


RUN mkdir -p /etc/ocdocker /workspace /opt/ocdocker/bin
COPY containers/docker/OCDocker.cfg.docker /etc/ocdocker/OCDocker.cfg.postgresql
COPY containers/docker/OCDocker.cfg.docker.mysql /etc/ocdocker/OCDocker.cfg.mysql
COPY containers/docker/entrypoint.sh /usr/local/bin/ocdocker-container-entrypoint
RUN chmod +x /usr/local/bin/ocdocker-container-entrypoint

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/ocdocker-container-entrypoint"]
CMD ["bash"]
