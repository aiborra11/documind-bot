#!/bin/bash

echo "Initializing Gunicorn for FastAPI..."

# Get the directory where this script is located (project root)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

exec gunicorn app.app:app \
    --chdir "$DIR" \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300