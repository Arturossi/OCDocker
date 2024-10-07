# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Install necessary dependencies
RUN apt update && apt install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Predefined directory inside the container for Vina
WORKDIR /opt/vina_split

# Build argument for the external source directory
ARG VINA_SPLIT_SOURCE

# Copy the vina_split source from outside (dynamic path)
COPY ${VINA_SPLIT_SOURCE} /opt/vina_split/

# Set environment variables to include Vina in the PATH
ENV PATH="/opt/vina:${PATH}"

# Set a default command (optional)
CMD ["vina_split"]
