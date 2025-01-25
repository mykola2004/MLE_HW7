import os
import json
import torch
import pandas as pd
import logging
from torch import nn
from datetime import datetime
from typing import List
import time

# Configuring logging
logging.basicConfig(level=logging.INFO)

# Loading configuration
CONF_FILE = os.getenv("CONF_PATH", "settings.json")
with open(CONF_FILE, "r") as f:
    conf = json.load(f)

DATA_DIR = conf["general"]["data_dir"]
MODEL_DIR = conf["general"]["models_dir"]
RESULTS_DIR = conf["general"]["results_dir"]
INFER_FILE = os.path.join(DATA_DIR, conf["inference"]["inp_table_name"])

# Neural network class from training file(same architecture)
class NeuralNet(nn.Module):
    def __init__(self, input_size,num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Defining function that loads model
def load_model(model_path: str, input_size: int, num_classes: int) -> nn.Module:
    try:
        model = NeuralNet(input_size, num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

# Defining function that loads inference data
def load_data(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded from {data_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Defining function that performs inference
def perform_inference(model: nn.Module, data: pd.DataFrame) -> pd.DataFrame:
    try:
        start_time = time.time()
        dataset_size = len(data)

        inputs = torch.tensor(data.values, dtype=torch.float32)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        data["predictions"] = predictions.numpy()

        time_spent = time.time() - start_time
        logging.info(
            f"Inference completed: Dataset Size: {dataset_size} samples, "
            f"Time Spent: {time_spent:.2f}s"
        )
        
        return data
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise

# Defining function that saves results
def save_results(results: pd.DataFrame, results_dir: str):
    try:
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime(conf["general"]["datetime_format"])
        results_path = os.path.join(results_dir, f"inference_results_{timestamp}.csv")
        results.to_csv(results_path, index=False)
        logging.info(f"Results saved to {results_path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def main():
    # Loading saved model
    model_path = os.path.join(MODEL_DIR, conf["inference"]["model_name"])
    model = load_model(model_path, input_size=4, num_classes=3)

    # Loading inference data
    inference_data = load_data(INFER_FILE)

    # Performing inference
    results = perform_inference(model, inference_data)

    # Saving inference results
    save_results(results, RESULTS_DIR)

if __name__ == "__main__":
    main()