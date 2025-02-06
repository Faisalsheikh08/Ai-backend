from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the Chrome extension

try:
    with open("./final_model.sav", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    model = None
    print("Error: Model file not found. Ensure 'final_model.sav' is in the correct path.")

@app.route("/")
def home():
    return "Fake News Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data["text"].strip()  # Extract only the text content
    if not text:
        return jsonify({"error": "Empty 'text' field in request"}), 400
    
    prediction = model.predict([text])
    
    return jsonify({"prediction": "Fake" if prediction[0] else "Real"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Allow external access
