# BIRD-Interact reproduction package — eval container
#
# Minimal eval image: Python 3.10 + postgresql-client + project deps.
# Does NOT contain cluster-ops tooling (kubectl, oidc-login, TeX, etc.) —
# those were specific to our cluster setup and aren't needed for reproduction.

FROM python:3.10

WORKDIR /app

# PostgreSQL client + git (sometimes needed by pip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Project dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Project tree
COPY src/  /app/src/
COPY data/ /app/data/
COPY scripts/ /app/scripts/
COPY pyproject.toml /app/

RUN pip install -e . --no-deps

# Stay alive so a user can `docker exec -it ... bash` and run scripts
CMD ["tail", "-f", "/dev/null"]
