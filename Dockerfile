FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS builder

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv "$VIRTUAL_ENV"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=root:root . .
COPY --chown=root:root healthcheck.py /usr/local/bin/healthcheck.py
RUN addgroup --gid 10001 appgroup \
    && adduser --uid 10001 --gid 10001 --disabled-password --gecos "" appuser \
    && mkdir -p /app/data /home/appuser \
    && chown -R appuser:appuser /app/data /home/appuser \
    && chmod -R a=rX /app \
    && chmod -R u=rwX,go=rX /app/data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python /usr/local/bin/healthcheck.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]