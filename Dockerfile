FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY pyproject.toml .
COPY evolvagent/ evolvagent/
COPY config.toml .

RUN pip install --no-cache-dir ".[network]"

# Data directory
RUN mkdir -p /data/logs
ENV EVOLVAGENT_DATA_DIR=/data

EXPOSE 8765

CMD ["evolvagent", "repl"]
