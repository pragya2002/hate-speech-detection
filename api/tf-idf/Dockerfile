FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY bot.py .
COPY model_weights.json .
COPY stopwords.json .

EXPOSE 5000

# Single worker, 4 threads — model is in-memory, thread-safe numpy ops
CMD ["gunicorn", "--workers=1", "--threads=4", "--timeout=30", "--bind=0.0.0.0:5000", "app:app"]
