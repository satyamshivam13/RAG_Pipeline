#!/usr/bin/env sh
set -e

# Start script used as Docker ENTRYPOINT. Runs the provided CMD as the app user.
# This script allows the container to be debugged by overriding CMD at runtime.

# If running database migrations or other startup tasks, add them here before exec.

# Ensure vector store dir exists with correct permissions
mkdir -p /vector_store_data
chown -R $(id -u):$(id -g) /vector_store_data || true

exec "$@"
