FROM python:3.11-slim

# Work directory inside the container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Install any additional system tools you need (e.g. nano)
RUN apt-get update && \
    apt-get install -y build-essential libopenblas-dev git nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# Copy your application code into /app
# COPY . /app

# Default entrypoint: runs "python <script>"
ENTRYPOINT ["python"]
# CMD ["main.py"]