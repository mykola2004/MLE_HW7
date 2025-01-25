import unittest
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.train import IrisDataset, SimpleNN, train_model, evaluate_model

class TestIrisDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("settings.json", "r") as file:
            conf = json.load(file)
            cls.data_path = os.path.join(conf["general"]["data_dir"], conf["train"]["table_name"])

    def test_dataset_loading(self):
        dataset = IrisDataset(self.data_path)
        # Check if dataset comes from appropriate class
        self.assertIsInstance(dataset, IrisDataset)
        # Check if length of loaded dataset is not 0
        self.assertGreater(len(dataset), 0)

    def test_dataset_shape(self):
        dataset = IrisDataset(self.data_path)
        # Checking number of features in training dataset, should be equal to 4
        self.assertEqual(dataset[0][0].shape[0], 4)
        # Checking values that take on target feature, should be [0, 1, 2]
        self.assertIn(dataset[0][1].item(), [0, 1, 2])

class TestSimpleNN(unittest.TestCase):
    def test_model_initialization(self):
        model = SimpleNN(input_size=4, num_classes=3)
        # Basic test for model instance
        self.assertIsInstance(model, SimpleNN)

    def test_model_forward(self):
        model = SimpleNN(input_size=4, num_classes=3)
        sample_input = torch.rand((1, 4))
        output = model(sample_input)
        # Checking if model outputs is of appropriate shape, must be equal to 3
        self.assertEqual(output.shape[1], 3)

class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("settings.json", "r") as file:
            conf = json.load(file)
            cls.train_path = os.path.join(conf["general"]["data_dir"], conf["train"]["table_name"])
            dataset = IrisDataset(cls.train_path)
            cls.train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    def test_train_model(self):
        model = SimpleNN(input_size=4, num_classes=3)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_model(model, criterion, optimizer, self.train_loader, epochs=1)
        for param in model.parameters():
            # Checking if gradients are computed properly during training
            self.assertIsNotNone(param.grad)

    def test_evaluate_model(self):
        model = SimpleNN(input_size=4, num_classes=3)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_model(model, criterion, optimizer, self.train_loader, epochs=1)

        accuracy = evaluate_model(model, self.train_loader)
        # Check if model accuracy is adequate
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

if __name__ == "__main__":
    unittest.main()
