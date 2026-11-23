FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENV LOKY_MAX_CPU_COUNT=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
# Downloads MovieLens and trains at build time so the container starts instantly (needs network).
RUN python train.py && useradd -m app && chown -R app /app
USER app
EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8050/')"
# Dashboard by default. For the API instead:
#   docker run -p 8000:8000 movie-studio uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["gunicorn", "dashboard:server", "-b", "0.0.0.0:8050", "--workers", "2"]
