#!/usr/bin/env sh
set -e

# Start script used as Docker ENTRYPOINT. Runs the provided CMD as the app user.
# This script allows the container to be debugged by overriding CMD at runtime.

# If running database migrations or other startup tasks, add them here before exec.

# Ensure vector store dir exists with correct permissions
# Use the application working directory to match `config.py` and the Dockerfile.
mkdir -p /app/vector_store_data
# Ownership is set at image build time; avoid aggressive chown at container start.

# If no command provided, run gunicorn with optional GUNICORN_WORKERS env var.
if [ "$#" -eq 0 ]; then
	GWORKERS=${GUNICORN_WORKERS:-4}
	exec sh -c "gunicorn -k uvicorn.workers.UvicornWorker -w ${GWORKERS} --bind 0.0.0.0:8000 api:app --access-logfile - --log-level info"
else
	exec "$@"
fi
