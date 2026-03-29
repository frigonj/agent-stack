FROM python:3.11-slim

WORKDIR /agent

# Copy local Emrys source (cloned via setup.sh) and install
COPY emrys-src/ /tmp/emrys-src/
RUN pip install --no-cache-dir "/tmp/emrys-src[vectors]" && \
    rm -rf /tmp/emrys-src

RUN mkdir -p /agent/.persist

ENV EMRYS_PERSIST_DIR=/agent/.persist

# Healthcheck: confirm the persist directory is accessible
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD test -d /agent/.persist || exit 1

# Keep the volume alive and run journal rotation every hour
CMD ["sh", "-c", "while true; do emrys rotate --persist-dir /agent/.persist 2>/dev/null || true; sleep 3600; done"]
