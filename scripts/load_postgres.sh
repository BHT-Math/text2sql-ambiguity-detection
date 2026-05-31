#!/usr/bin/env bash
# Pull the pre-built BIRD-Interact lite postgres image and start it locally.
#
# The image bundles all 20 lite-300 databases with their schemas + data
# pre-loaded. No SQL dumps to import; the image is the ground truth.
#
# Usage: ./scripts/load_postgres.sh
#
# After this finishes you should see a healthy container on localhost:5432
# with POSTGRES_USER=root POSTGRES_PASSWORD=123123.

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
  echo "ERROR: docker-compose (or 'docker compose') not found." >&2
  exit 1
fi

# Prefer the v2 plugin if available, fall back to legacy binary.
if docker compose version &> /dev/null; then
  COMPOSE="docker compose"
else
  COMPOSE="docker-compose"
fi

echo "→ Starting postgres service…"
$COMPOSE -f docker/docker-compose.yml up -d postgres

echo "→ Waiting for postgres to become healthy (up to 60s)…"
for _ in $(seq 1 30); do
  if docker exec bird_interact_postgres pg_isready -U root &> /dev/null; then
    echo "✓ Postgres ready at localhost:5432 (user=root, password=123123)"
    exit 0
  fi
  sleep 2
done

echo "ERROR: postgres did not become ready in 60s. Check logs:" >&2
echo "  docker logs bird_interact_postgres" >&2
exit 1
