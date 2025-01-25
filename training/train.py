import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score

# Configuring logging
logging.basicConfig(level=logging.INFO)

# Loading configuration fron specific file setting.json
CONF_PATH = os.getenv("CONF_PATH", "settings.json")
with open(CONF_PATH, "r") as f:
    conf = json.load(f)

# Getting directory name where input data located and dircetory name where trained model should be stored
DATA_PATH = conf["general"]["data_dir"]
MODEL_DIR = conf["general"]["models_dir"]
TRAIN_FILE = os.path.join(DATA_PATH, conf["train"]["table_name"])

# Checking if directory for saving model exists, if not creating it
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Dataset class definition
class IrisDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.x = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# Neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, criterion, optimizer, train_loader, epochs):
    try: 
        model.train()
        dataset_size = len(train_loader.dataset)
        for epoch in range(epochs):
            total_loss = 0
            start_time = time.time()
            for x, y in train_loader:
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            time_spent = time.time() - start_time
            logging.info(
                f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, "
                f"Time Spent: {time_spent:.2f}s, Dataset Size: {dataset_size} samples"
            )
    except Exception as e:
        logging.error(f"Error while training model: {e}")

# Evaluation function
def evaluate_model(model, test_loader):
    try: 
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(y.numpy())
        acc = accuracy_score(all_labels, all_preds)
        logging.info(f"Test Accuracy: {acc:.4f}")
        return acc
    except Exception as e:
        logging.error(f"Error while evaluating model: {e}")

# Saving model function
def save_model(model, path):
    try:
        torch.save(model.state_dict(), path)
        logging.info(f"Model saved to {path}")
    except Exception as e: 
        logging.error(f"Error saving model: {e}")

def main():
    # Data preparing
    try: 
        dataset = IrisDataset(TRAIN_FILE)
    except Exception as e: 
        logging.error(f"Error downloading training dataset: {e}")

    train_size = int(len(dataset) * conf["train"]["train_ratio"])
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=conf["train"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=conf["train"]["batch_size"], shuffle=False)

    # Defining input size; defining number of features that each samle in dataset has -- 4
    input_size = dataset[0][0].shape[0]
    # Defining number of labels of target feature -- 3
    num_classes = len(set(dataset.y))
    # Defining model, criterion for optimization, optimization algorithm
    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=conf["train"]["learning_rate"])

    # Training and evaluating model
    train_model(model, criterion, optimizer, train_loader, conf["train"]["epochs"])
    evaluate_model(model, test_loader)

    # Saving trained model to specified path in root directory, which is ./models
    model_path = os.path.join(MODEL_DIR, "trained_model.pickle")
    save_model(model, model_path)

if __name__ == "__main__":
    main()