# Local development / self-hosting image.
#
# Render builds from requirements.txt directly (see render.yaml) and does not
# use this file - the webhook service must stay small enough for the free tier.
# This image includes rembg so a single container can also do the inference,
# which is useful on a VPS or for testing without GitHub Actions.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REMBG_HOME=/models

WORKDIR /app

COPY requirements.txt requirements-worker.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-worker.txt

COPY . .

# Model cache survives container restarts when mounted as a volume.
VOLUME ["/models"]

EXPOSE 10000

CMD ["python", "app.py"]
