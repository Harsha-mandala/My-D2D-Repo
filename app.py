import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file, redirect, url_for
from ultralytics import YOLO
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
yolo_model = YOLO("C:/Users/Venu Gopal/Desktop/Files/runs/detect/train6/weights/best.pt")  # Use your latest best.pt
cnn_model = load_model("C:/Users/Venu Gopal/Desktop/Files/traffic_sign_cnn_model.h5")

# Example: Your class names (update with your actual class names)
class_names = {
    0: "speed limit 20 (prohibitory)",
    1: "speed limit 30 (prohibitory)",
    2: "speed limit 50 (prohibitory)",
    3: "speed limit 60 (prohibitory)",
    4: "speed limit 70 (prohibitory)",
    5: "speed limit 80 (prohibitory)",
    6: "restriction ends 80 (other)",
    7: "speed limit 100 (prohibitory)",
    8: "speed limit 120 (prohibitory)",
    9: "no overtaking (prohibitory)",
    10: "no overtaking (trucks) (prohibitory)",
    11: "priority at next intersection (danger)",
    12: "priority road (other)",
    13: "give way (other)",
    14: "stop (other)",
    15: "no traffic both ways (prohibitory)",
    16: "no trucks (prohibitory)",
    17: "no entry (other)",
    18: "danger (danger)",
    19: "bend left (danger)",
    20: "bend right (danger)",
    21: "bend (danger)",
    22: "uneven road (danger)",
    23: "slippery road (danger)",
    24: "road narrows (danger)",
    25: "construction (danger)",
    26: "traffic signal (danger)",
    27: "pedestrian crossing (danger)",
    28: "school crossing (danger)",
    29: "cycles crossing (danger)",
    30: "snow (danger)",
    31: "animals (danger)",
    32: "restriction ends (other)",
    33: "go right (mandatory)",
    34: "go left (mandatory)",
    35: "go straight (mandatory)",
    36: "go right or straight (mandatory)",
    37: "go left or straight (mandatory)",
    38: "keep right (mandatory)",
    39: "keep left (mandatory)",
    40: "roundabout (mandatory)",
    41: "restriction ends (overtaking) (other)",
    42: "restriction ends (overtaking (trucks)) (other)"
}


# Create uploads folder if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))
    if file:
        # Save the uploaded file
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        # Read and process the image
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # YOLO detection
        results = yolo_model(img_rgb)
        
        # For each detected traffic sign
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                class_id = int(box.cls)
                # Crop the detected sign
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # Preprocess for CNN
                crop_resized = cv2.resize(crop, (32, 32))
                crop_normalized = crop_resized.astype("float32") / 255.0
                crop_input = np.expand_dims(crop_normalized, axis=0)
                # CNN classification
                predictions = cnn_model.predict(crop_input)
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions)
                # Draw results
                label = f"{class_names.get(predicted_class, 'Unknown')} {confidence:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the result
        result_path = os.path.join("uploads", "result.jpg")
        cv2.imwrite(result_path, img)
        
        # Display the result
        return render_template("index.html", result_image="result.jpg")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_file(os.path.join("uploads", filename))

if __name__ == "__main__":
    app.run(debug=True)
