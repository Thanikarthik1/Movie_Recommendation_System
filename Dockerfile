FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Update system packages to patch vulnerabilities
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy your files
COPY . .

# Install requirements
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Expose port and run
EXPOSE 7860
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "7860"]