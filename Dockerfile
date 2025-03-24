FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
RUN pip install --no-cache-dir \
    aiohttp==3.8.5 \
    numpy==1.24.3 \
    opencv-python==4.8.0.74 \
    mediapipe
    dotenv

# Copy application files
COPY app.py /app/server.py
COPY index.html /app/index.html

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "server.py"]