# ==========================================
#  Dockerfile for Trading Project
#  Python 3.11.9 + Flask
# ==========================================

# ---- Base image ----
FROM python:3.11-slim

# ---- Set working directory ----
WORKDIR /app

# ---- Install system dependencies if needed ----
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy requirements first and install dependencies ----
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Copy all project files ----
COPY . .

# ---- Create necessary directories ----
RUN mkdir -p /app/data

# ---- Expose Flask default port ----
EXPOSE 5000

# ---- Run the Flask app ----

CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "120", "--workers", "1", "--threads", "2", "app:app"]






