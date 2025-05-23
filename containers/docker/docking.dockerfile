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
    curl \
    jq \
    tar \
    && rm -rf /var/lib/apt/lists/*

#########################################
# AutoDock Vina and Vina Split Installation
#########################################

# Create a self-contained shell script to fetch and extract the binaries
RUN script_content='#!/bin/bash\n\
set -e\n\
\n\
# Fetch the latest release version\n\
LATEST_RELEASE=$(curl -s https://api.github.com/repos/ccsb-scripps/AutoDock-Vina/releases/latest | jq -r .tag_name)\n\
LATEST_RELEASE_SHORT=${LATEST_RELEASE:1}\n\
\n\
# Print the latest release information\n\
echo "Latest release: $LATEST_RELEASE"\n\
echo "Latest release short: $LATEST_RELEASE_SHORT"\n\
\n\
# Download Vina and Vina Split\n\
echo "Downloading Vina from: https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/${LATEST_RELEASE}/vina_${LATEST_RELEASE_SHORT}_linux_x86_64"\n\
wget -O vina https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/${LATEST_RELEASE}/vina_${LATEST_RELEASE_SHORT}_linux_x86_64\n\
\n\
echo "Downloading Vina Split from: https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/${LATEST_RELEASE}/vina_split_${LATEST_RELEASE_SHORT}_linux_x86_64"\n\
wget -O vina_split https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/${LATEST_RELEASE}/vina_split_${LATEST_RELEASE_SHORT}_linux_x86_64\n\
\n\
# Move the binaries to the appropriate directory\n\
mv vina /opt\n\
mv vina_split /opt\n\
' && echo "$script_content" > /fetch_and_extract.sh

# Make the script executable and run it
RUN chmod +x /fetch_and_extract.sh && /fetch_and_extract.sh

# Make vina and vina_split executables
RUN chmod +x /opt/vina /opt/vina_split

#########################################
# SMINA Installation
#########################################

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

# Clean up the temporary build directory
RUN rm -rf /tmp/smina

#########################################
# Set up the PATH and Entry Point
#########################################

# Add all binaries to the PATH
ENV PATH="/opt:${PATH}"

# Create the entry point script to allow choosing between vina, vina_split, and smina
RUN echo '#!/bin/bash\n\
\n\
# Check if a command is provided\n\
if [ "$#" -eq 0 ]; then\n\
    echo "Usage: $0 {vina|vina_split|smina} [args]"\n\
    exit 1\n\
fi\n\
\n\
# Run the specified command with the provided arguments\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Set the entry point to the script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
