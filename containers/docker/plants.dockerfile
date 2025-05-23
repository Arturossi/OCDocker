# Use an appropriate base image
FROM ubuntu:22.04

# Copy the PLANTS executable from the user's provided path into the container
ARG PLANTS_EXECUTABLE_PATH

# Copy the executable into /opt in the container
COPY ${PLANTS_EXECUTABLE_PATH} /opt/plants

# Make the binary executable
RUN chmod +x /opt/plants

# Set the PATH environment variable
ENV PATH="/opt:${PATH}"

# Create the entry point that directly runs the PLANTS executable
RUN echo '#!/bin/bash\n\
\n\
# Run the PLANTS binary with the provided arguments\n\
exec "/opt/plants" "$@"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Set the entry point to the script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
