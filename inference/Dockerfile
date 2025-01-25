# Base Python image
FROM python:3.10

# Argument for the model name and settings
ARG model_name=trained_model.pickle
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Set working directory
WORKDIR /app

# Copy data, models, and inference script, other scripts into the container
COPY data/ /app/data
COPY models/${model_name} /app/models/${model_name}
COPY inference/ /app/inference
COPY utils.py /app/
COPY ${CONF_PATH} /app/

# Install necessary dependencies, before that upgrade pip
COPY requirements_inference.txt .
RUN pip install --upgrade pip
# requiremnets file is different than that one used for training
RUN pip install --no-cache-dir -r requirements_inference.txt

# Run the inference script
CMD ["python3", "inference/run.py"]