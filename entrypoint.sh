#!/bin/sh
set -e

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs -d '\n')

# Execute the main command
exec "$@"