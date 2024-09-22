# YOLOv8 Military Object Detection Project

This project uses the YOLOv8 model for custom object detection on a dataset consisting of 6 object classes: `truck`, `car`, `tank`, `aircraft`, `aircraft fighter`, and `helicopter`. The model is trained using Google Colab and the dataset stored in Google Drive.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)

## Project Structure
## Installation

1. **Install YOLOv8**:
   First, install the `ultralytics` package for YOLOv8 in Colab:

   ```bash
   !pip install ultralytics
   from google.colab import drive
drive.mount('/content/drive', force_remount=True)
names: ["truck", "car", "tank", "aircraft", "aircraft fighter", "helicopter"]

Prepare the YAML File:
You need to create a data_custom.yaml file that defines the paths to your dataset and class labels:

data_yaml = """
train: drive/MyDrive/yolov_intern/train
val: drive/MyDrive/yolov_intern/val

nc: 6

names: ["truck", "car", "tank", "aircraft", "aircraft fighter", "helicopter"]
"""

# Save the string to a .yaml file
with open('/content/data_custom.yaml', 'w') as file:
    file.write(data_yaml)

    Train the Model:
The pre-trained yolov8m.pt weights are loaded from Google Drive, and the model is trained using the custom dataset:

from ultralytics import YOLO

# File path to the pre-trained model
file_path = "/content/drive/MyDrive/yolov_intern"

# Load the pre-trained YOLOv8 model
model = YOLO(f"{file_path}/yolov8m.pt")

# Train the model
model.train(data="/content/data_custom.yaml", batch=8, imgsz=640, epochs=100, workers=1, patience=5)

Training Parameters:

	•	batch: 8
	•	imgsz: 640
	•	epochs: 100
	•	workers: 1
	•	patience: 5

Evaluation

Once the model is trained, you can evaluate its performance using the validation set. YOLOv8 provides built-in functions to evaluate the model on different metrics such as precision, recall, and mAP.


results = model.val()
print(results)

Results

After training, the model’s performance will be evaluated based on the dataset. You can visualize the model’s predictions on the validation set using:

model.predict(source='/content/drive/MyDrive/yolov_intern/val', save=True)
