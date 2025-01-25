## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
MLE_basic_example
├── data                       # Data files used for training and inference
│   ├── iris_inference.csv
│   └── iris_train.csv         
├── inference                  # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── training                   # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── unittests                  # Scripts for performing unit tests of main parts of project
│   └──unittests.py
├── utils.py                   # Utility functions and classes that are used in scripts
├── settings.json              # All configurable parameters and settings
├── requirements.txt           # File listing all necessary libraries used for training
├── requirements_inference.txt # File listing all necessary libraries used for inference        
└── README.md                  # Descriprion of the project, instruction how to launch a project
```

## Settings:
The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project. Examples could be the path to certain resource files, constant values, hyperparameters for an ML model, or specific settings for different environments. Before running the project, ensure that all the paths and parameters in `settings.json` are correctly defined.
Keep in mind that you may need to pass the path to your config to the scripts. For this, you may create a .env file and manually initialize an environment variable as `CONF_PATH=settings.json`.
Please note, some IDEs, including VSCode, may have problems detecting environment variables defined in the .env file. This is usually due to the extension handling the .env file. If you're having problems, try to run your scripts in a debug mode, or, as a workaround, you can hardcode necessary parameters directly into your scripts. Make sure not to expose sensitive data if your code is going to be shared or public. In such cases, consider using secret management tools provided by your environment.

## Data:
Data is already stored in folder "/data", it is prepared for training and inferencing.

## Training:
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.

To train the model using Docker: 

- Build the training Docker image.
```bash
docker build -f training/Dockerfile -t training_image .
```
- Then run the container(that will trigger: training model, printing all logs related to model's perfomance during training, saving model inside container):
```bash
docker run training_image
```
- Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using command:
```bash
docker cp <container_id>:/app/models ./models
```
Replace `<container_id>` with your running Docker container ID.
After that ensure that you have your model saved in the `/models` directory.

## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

To run the inference using Docker, use the following commands:

- Build the inference Docker image from appropriate docker file:
```bash
docker build -f inference/Dockerfile -t inference_image .
```
- Run the inference Docker container(that will cause model to predict values on unseen during data - inference dataset, the results will be saved inside container):
```bash
docker run inference_image
```
- Then, move saved predictions file from container to your local machine with the command:
```bash
docker cp <container_id>:/app/results ./results
```
Replace `<container_id>` with your running Docker container ID.
After that ensure that you have your results in the `/results` directory.