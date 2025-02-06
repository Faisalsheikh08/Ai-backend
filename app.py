from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)

# Allow specific domains
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://vite.dev"]}})  # Allow specific origins like localhost:5173

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

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # Handle preflight request
        response = jsonify({"message": "CORS preflight response OK"})
        response.status_code = 200
        # Send CORS headers
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"  # Allow origin
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"  # Allowed methods
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"  # Allowed headers
        return response

    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        # Log the incoming request data for debugging
        data = request.get_json()
        print(f"Received data: {data}")

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Empty 'text' field in request"}), 400

        # Perform prediction
        prediction = model.predict([text])
        result = "Fake" if prediction[0] else "Real"
        return jsonify({"prediction": result})

    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Set debug=True for development
