from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import pickle
import pandas as pd

# Configure logging to print everything for debugging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
try:
    model = pickle.load(open('classifier.pkl', 'rb'))
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Use the exact column names from your dataset
column_names = [
    "Temparature", "Humidity ", "Moisture", "Soil Type", "Crop Type",
    "Nitrogen", "Potassium", "Phosphorous"
]  # Note: "Humidity " has a space

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global sensor data
sensor_data = {"temperature": 0, "humidity": 0, "soil_moisture": 0}
irrigation_state = False  # Track irrigation state (False = OFF, True = ON)

@app.route('/get_irrigation_state', methods=['GET'])
def get_irrigation_state():
    logging.debug("GET /get_irrigation_state called")
    return jsonify({"irrigation_state": irrigation_state})

@app.route('/set_irrigation_state', methods=['POST'])
def set_irrigation_state():
    global irrigation_state
    data = request.get_json()
    logging.debug(f"POST /set_irrigation_state received: {data}")

    if data and "state" in data:
        irrigation_state = data["state"]
        return jsonify({"success": True, "irrigation_state": irrigation_state})
    return jsonify({"success": False, "message": "Invalid request"}), 400

@app.route('/sensor', methods=['POST'])
def sensor_data_update():
    global sensor_data
    try:
        data = request.json
        logging.debug(f"POST /sensor received: {data}")

        if not all(key in data for key in ["temperature", "humidity", "soil_moisture"]):
            return jsonify({"status": "error", "message": "Missing required sensor data!"}), 400
        
        sensor_data.update(data)
        logging.info(f"Updated sensor data: {sensor_data}")
        return jsonify({"status": "success", "message": "Sensor data received"}), 200
    except Exception as e:
        logging.error(f"Error updating sensor data: {e}")
        return jsonify({"status": "error", "message": "Invalid data"}), 400

@app.route('/get_sensor_data', methods=['GET'])
def get_sensor_data():
    logging.debug("GET /get_sensor_data called")
    return jsonify(sensor_data), 200

@app.route("/process_data", methods=["POST"])
def process_data():
    try:
        data = request.get_json()
        logging.debug(f"POST /process_data received: {data}")

        # Extracting input values
        nitrogen = data.get("nitrogen")
        potassium = data.get("potassium")
        phosphorous = data.get("phosphorous")
        soil_type = data.get("soilType")
        crop_type = data.get("cropType")

        # Encoding mappings
        soil_mapping = {"Sandy": 0, "Loamy": 1, "Clayey": 2, "Peaty": 3, "Saline": 4, "Chalky": 5, "Silty": 6}
        crop_mapping = {"Wheat": 0, "Rice": 1, "Maize": 2, "Barley": 3, "Sugarcane": 4, "Cotton": 5, "Vegetables": 6}

        soil_encoded = soil_mapping.get(soil_type, -1)
        crop_encoded = crop_mapping.get(crop_type, -1)

        if soil_encoded == -1 or crop_encoded == -1:
            logging.error(f"Invalid soil or crop type: Soil-{soil_type}, Crop-{crop_type}")
            return jsonify({"status": "error", "message": "Invalid Soil or Crop Type"}), 400

        # Formulating input for the model
        sample_input = [[
            sensor_data["temperature"], sensor_data["humidity"], sensor_data["soil_moisture"],
            soil_encoded, crop_encoded, nitrogen, potassium, phosphorous
        ]]
        logging.debug(f"Model Input: {sample_input}")

        # Convert input to DataFrame
        sample_df = pd.DataFrame(sample_input, columns=column_names)

        # Predict the Fertilizer Type
        prediction = model.predict(sample_df)
        logging.info(f"Model Prediction: {prediction}")

        # Fertilizer Mapping
        fertilizer_mapping = {0: "10-26-26", 1: "14-35-14", 2: "17-17-17", 3: "20-20", 4: "28-28", 5: "DAP", 6: "Urea"}

        # Return Prediction
        return jsonify({"Predicted Fertilizer": fertilizer_mapping.get(prediction[0], "Unknown")})

    except Exception as e:
        logging.error(f"Error in /process_data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    logging.info("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
