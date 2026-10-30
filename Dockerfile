FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY --chown=app:app models/best_model.joblib ./models/best_model.joblib

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "house_price.api:app", "--host", "0.0.0.0", "--port", "8000"]
