

# Use an appropriate base image
FROM ubuntu:22.04

# Install required packages
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

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
# Download Vina Split\n\
\n\
echo "Downloading Vina Split from: https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/${LATEST_RELEASE}/vina_split_${LATEST_RELEASE_SHORT}_linux_x86_64"\n\
wget -O vina_split https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/${LATEST_RELEASE}/vina_split_${LATEST_RELEASE_SHORT}_linux_x86_64\n\
\n\
# Move the binaries to the appropriate directory\n\
mv vina_split /opt\n\
' && echo "$script_content" > /fetch_and_extract.sh

# Make the script executable
RUN chmod +x /fetch_and_extract.sh

# Run the script
RUN /fetch_and_extract.sh

# Make vina and vina_split executable
RUN chmod +x /opt/vina_split

# Set the PATH
ENV PATH="/opt:${PATH}"

# Create the entry point script to allow choosing between vina and vina_split
RUN echo '#!/bin/bash\n\
\n\

# Run the specified command with the provided arguments\n\
exec "vina_split"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Set the entry point to the script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
