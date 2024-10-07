# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    cmake \
    libboost-all-dev \
    libopenbabel-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set a temporary working directory for building SMINA
WORKDIR /tmp/smina

# Clone the SMINA repository
RUN git clone https://git.code.sf.net/p/smina/code smina-code

# Build SMINA
RUN cd smina-code && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) smina

# Move the SMINA executable to /opt and make it executable
RUN mv /tmp/smina/smina-code/build/smina /opt/smina && chmod +x /opt/smina

# Set the final working directory
WORKDIR /opt

# Clean up the temporary build directory
RUN rm -rf /tmp/smina

# Add the SMINA executable to the PATH
ENV PATH="/opt:${PATH}"

# Create the entry point script directly in the Dockerfile
RUN echo '#!/bin/bash\n\
\n\
# Run SMINA with the provided arguments\n\
exec /opt/smina "$@"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Set the entry point to the script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
