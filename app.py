from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import pickle
from PIL import Image
from io import BytesIO
import requests
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten

app = Flask(__name__)

# Load trained model and scaler
try:
    model = pickle.load(open("svm_model_new.pkl", "rb"))
    print("Model Loaded Successfully")
    scaler = pickle.load(open("scalerr.pkl", "rb"))
    print("Scaler Loaded Successfully")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Load VGG16 model without pooling, and add a flatten layer.
try:
    vgg_model = VGG16(weights='imagenet', include_top=False)
    flatten = Flatten()
    print("VGG Model Loaded Successfully")
except Exception as e:
    print(f"Error loading VGG model: {e}")
    vgg_model = None
    flatten = None

# ESP32-CAM IP Address (Replace with your ESP32-CAM URL)
ESP32_CAM_URL = "http://10.0.1.22/capture"

# Upload folder
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

EXPECTED_SIZE = (224, 224)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=["GET"])
def capture():
    try:
        response = requests.get(ESP32_CAM_URL)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img_path = os.path.join(UPLOAD_FOLDER, "captured_leaf.jpg")
            img.save(img_path)
            return jsonify({"success": True, "img_path": img_path})
        else:
            return jsonify({"success": False, "error": "Failed to capture image"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        return jsonify({"success": True, "img_path": img_path})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        img_path = data.get("img_path")

        if not img_path or not os.path.exists(img_path):
            return jsonify({"success": False, "error": "Image path is invalid"}), 400

        if model is None or scaler is None or vgg_model is None or flatten is None:
            return jsonify({"success": False, "error": "Model or scaler not loaded"}), 500

        img = image.load_img(img_path, target_size=EXPECTED_SIZE)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        vgg_features = vgg_model.predict(img)
        vgg_features_flattened = flatten(vgg_features).numpy()

        img_scaled = scaler.transform(vgg_features_flattened)

        prediction = model.predict(img_scaled)[0]
        print(f"Raw Prediction: {prediction}")

        prediction_map = {
            "brinjal healthy": {"label": "Healthy","raw":0},
            "brinjaldiseased": {"label": "Diseased","raw":0},
            "beans diseased":{"label": "Diseased","raw":1},
            "bg diseased":{"label":"Diseased","raw":1},
            "bottleguarddiseased":{"label":"Diseased","raw":1},
            "beans diseased":{"label":"Diseased","raw":1},
            "jungleflowerdiseased":{"label":"Diseased","raw":1},
            "beans healthy":{"label":"Healthy","raw":0},
            "black gram healthy":{"label":"Healthy","raw":0},
            "bottleguard healthy":{"label":"Healthy","raw":0},
            "guava healthy":{"label":"Healthy","raw":0},
            "hibiscus healthy":{"label":"Healthy","raw":0},
            "jungle flower healthy":{"label":"Healthy", "raw":0}
        }

        if prediction in prediction_map:
            label = prediction_map[prediction]["label"]
            raw_prediction = prediction_map[prediction]["raw"]
            return jsonify({"success": True, "label": label, "raw_prediction": raw_prediction})
        else:
            return jsonify({"success": False, "error": f"Unknown prediction: {prediction}"}), 500

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/test")
def test():
    return "Flask is working!"

if __name__ == "__main__":
    app.run(debug=True, port=5000)