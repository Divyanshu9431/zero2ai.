from ultralytics import YOLO
import os

model = YOLO("yolov8m_custom.pt")

file_dir = './train/images'

for file in os.listdir(file_dir):
    file_path = os.path.join(file_dir, file)
    
    if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        model.predict(source=file_path, show=True, save=True, conf=0.5)
