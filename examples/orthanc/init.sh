#!/bin/sh
# Creates .env and ./credentials with random secrets for the example.
# Existing files are kept, so running it twice does not rotate secrets.
set -eu
cd "$(dirname "$0")"
if [ -e .env ] || [ -e credentials ]; then
  echo ".env or credentials already exists; delete them to start over" >&2
  exit 1
fi
umask 077
key="tomoz$(openssl rand -hex 6)"
secret="$(openssl rand -hex 32)"
printf '%s:%s\n' "$key" "$secret" > credentials
# Compose bind-mounts the file with its host permissions, and the gateway runs
# as uid 10001 in its container, so the file must be readable by others. Use
# your orchestrator's secrets (with ownership) in production.
chmod 644 credentials
cat > .env <<ENV
TOMOZ_ACCESS_KEY=$key
TOMOZ_SECRET_KEY=$secret
ORTHANC_PASSWORD=$(openssl rand -hex 16)
ENV
echo "wrote .env and credentials (keep them private)"
