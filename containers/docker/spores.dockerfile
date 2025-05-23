# Use an appropriate base image
FROM ubuntu:22.04

# Create an entry point that runs the SPORES binary from the mounted path
RUN echo '#!/bin/bash\n\
\n\
# Check if a path is provided\n\
if [ -z "$SPORES_PATH" ]; then\n\
    echo "Error: SPORES_PATH environment variable is not set. Please provide the path to SPORES binary."\n\
    exit 1\n\
fi\n\
\n\
# Run the provided SPORES binary with the passed arguments\n\
exec "$SPORES_PATH/spores_linux" "$@"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Set the entry point to the script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
