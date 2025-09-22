FROM python:3.11-slim

# Базовые либы для opencv/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости (лучше кешируется)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY . .

ENV PYTHONUNBUFFERED=1

# Heroku запустит как worker
CMD ["python", "bot.py"]
