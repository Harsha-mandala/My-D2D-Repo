# My-D2D-Repo
# Traffic Sign Detection and Recognition System

This project implements a complete pipeline for detecting and recognizing traffic signs using deep learning. It combines YOLOv8 for traffic sign detection and a custom CNN for traffic sign classification, all served through a Flask web application.

## Table of Contents

- [1. Dataset Preparation](#1-dataset-preparation)
- [2. YOLOv8 Training (Detection)](#2-yolov8-training-detection)
- [3. CNN Training (Recognition)](#3-cnn-training-recognition)
- [4. Model Integration](#4-model-integration)
- [5. Flask Web Application](#5-flask-web-application)
- [6. Running the Project](#6-running-the-project)
- [7. Project Structure](#7-project-structure)
- [8. Requirements](#8-requirements)
- [9. Troubleshooting](#9-troubleshooting)

## 1. Dataset Preparation

- **Download and organize** the datasets for detection and recognition. I have used chinese traffic sign recognition database and german traffic sign detection data base.
- **Detection (YOLO):**
  - Place images in `train/images/` and `val/images/`.
  - Use the provided `gt.txt` file to generate YOLO-format `.txt` annotation files in `train/labels/` and `val/labels/`.
  - Convert `.ppm` images to `.jpg` as YOLOv8 does not support `.ppm`.
- **Recognition (CNN):**
  - Place cropped/sign images for classification in separate folders for training and testing.
  - Use annotation files to map each image to its class label.

## 2. YOLOv8 Training (Detection)

- **Install Ultralytics YOLO:**
  ```bash
  pip install ultralytics
  ```
- **Create a `traffic_signs.yaml` file** specifying train/val image paths and class names.
- **Train the YOLOv8 model:**
  ```python
  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")
  results = model.train(
      data="path/to/traffic_signs.yaml",
      epochs=32,  # or your chosen number
      imgsz=640
  )
  ```
- **Model weights** are saved in `runs/detect/trainX/weights/best.pt`.

## 3. CNN Training (Recognition)

- **Install TensorFlow and other dependencies:**
  ```bash
  pip install tensorflow opencv-python numpy matplotlib scikit-learn
  ```
- **Preprocess images:** Resize to 32x32, normalize, and one-hot encode labels.
- **Build and train the CNN:**
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
  # ... (see project code for full model)
  model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
  model.save("traffic_sign_cnn_model.h5")
  ```

## 4. Model Integration

- **Load YOLO and CNN models in Python:**
  ```python
  from ultralytics import YOLO
  from tensorflow.keras.models import load_model
  yolo_model = YOLO("runs/detect/trainX/weights/best.pt")
  cnn_model = load_model("traffic_sign_cnn_model.h5")
  ```
- **Detection pipeline:**
  - YOLO detects bounding boxes for traffic signs.
  - Each detected region is cropped, resized, and classified by the CNN.
  - Results are drawn on the original image.

## 5. Flask Web Application

- **Install Flask:**
  ```bash
  pip install flask
  ```
- **Create `app.py`** with endpoints for uploading images and displaying results.
- **Create `templates/index.html`** for the web interface.
- **Run the app:**
  ```bash
  python app.py
  ```
- **Access the web app:** [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 6. Running the Project

1. **Prepare datasets and annotations as described.**
2. **Train YOLO and CNN models.**
3. **Place trained models (`best.pt` and `traffic_sign_cnn_model.h5`) in your project folder.**
4. **Start the Flask app:**
   ```bash
   python app.py
   ```
5. **Open your browser and use the web interface to upload images and see detection/classification results.**

## 7. Project Structure

```
project-root/
├── runs/
│   └── detect/
│       └── trainX/
│           └── weights/
│               ├── best.pt
│               └── last.pt
├── traffic_sign_cnn_model.h5
├── yolov8n.pt
├── app.py
└── templates/
    └── index.html
```

## 8. Requirements

- Python 3.11
- TensorFlow (2.15+)
- Ultralytics YOLO
- OpenCV
- Flask
- Numpy, Matplotlib, Scikit-learn

## 9. Troubleshooting

- **No images detected:** Lower YOLO confidence threshold or check model paths.
- **ModuleNotFoundError:** Install missing packages with `pip install ...`.
- **TensorFlow not installing:** Use Python 3.11, not 3.12+.
- **Flask app not running:** Check Python installation and run from terminal, not notebook.

**Project by:** Mandala Harsha Yogananda Bharati 
**Location:** Secunderabad, Telangana, India  
**Date:** July 2025
