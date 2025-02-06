from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the Chrome extension

# Model file path
MODEL_PATH = "./final_model.sav"

# Load the model safely
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("❌ Error: Model file not found! Ensure 'final_model.sav' is in the correct path.")

@app.route("/")
def home():
    return jsonify({"message": "Fake News Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty 'text' field in request"}), 400
    
    try:
        prediction = model.predict([text])
        result = "Fake" if prediction[0] else "Real"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)  # Set debug=False for production
