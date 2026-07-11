# Pinned to Debian bookworm (12), not the unqualified "slim" tag -- that
# now resolves to Debian trixie (13), which Playwright's dependency
# installer doesn't recognize; it falls back to Ubuntu 20.04 package names
# (ttf-ubuntu-font-family, ttf-unifont) that don't exist on Debian and the
# build fails. bookworm is a Playwright-supported OS.
FROM python:3.11-slim-bookworm

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libnss3 libatk-bridge2.0-0 libxkbcommon0 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps chromium

COPY app ./app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
