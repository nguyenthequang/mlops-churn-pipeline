# Inference image for the churn API. Hugging Face Spaces (Docker SDK) expects
# the app to listen on port 7860.
FROM python:3.11-slim

WORKDIR /app

# Install only the lean runtime deps first so this layer caches across rebuilds.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code + the champion model (preprocessing pipeline is baked into the model).
COPY app.py .
COPY champion/model.joblib champion/model.joblib

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
