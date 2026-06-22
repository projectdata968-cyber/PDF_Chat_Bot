FROM python:3.10-slim

WORKDIR /app

# Install modern system packages required for OpenCV, PaddleOCR, and PDF decoding
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them securely
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all remaining backend files
COPY . .

# Force FastAPI to bind explicitly to port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
