# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /opt/mgltools

# Download and install MGLTools
RUN wget https://ccsb.scripps.edu/download/532/ -O mgltools_install.tar.gz --no-check-certificate \
    && mkdir -p mgltools \
    && tar -xvzf mgltools_install.tar.gz -C mgltools --strip-components=1 \
    && rm mgltools_install.tar.gz \
    && cd mgltools \
    && . ./install.sh

# Add mgltools/bin to PATH for convenience
ENV PATH="/opt/mgltools/bin:${PATH}"

# Create the entry point script directly in the Dockerfile
RUN echo '#!/bin/bash\n\
\n\
# Check the command and run the appropriate script\n\
case "$1" in\n\
    prepareligand)\n\
        shift # Remove the first argument\n\
        /opt/mgltools/bin/pythonsh /opt/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py "$@"\n\
        ;;\n\
    preparereceptor)\n\
        shift # Remove the first argument\n\
        /opt/mgltools/bin/pythonsh /opt/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py "$@"\n\
        ;;\n\
    *)\n\
        echo "Usage: docker run <container_name> prepareligand <args> | preparereceptor <args>"\n\
        exit 1\n\
        ;;\n\
esac' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Set the entry point to the script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
