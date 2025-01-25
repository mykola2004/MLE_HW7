# Base Python image
FROM python:3.10

# Initialize settings_name variable with a default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Set working directory
WORKDIR /app

# Copy data folder to the Docker container
COPY data/ /app/data

# Copy training script and files
COPY training/ /app/training

# Copy utilities and configuration
COPY utils.py /app/
COPY ${CONF_PATH} /app/

# Install dependencies from appropriate file
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Run the training script
CMD ["python3", "training/train.py"]
