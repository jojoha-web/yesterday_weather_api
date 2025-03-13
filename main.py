from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import numpy as np
import joblib
import tensorflow as tf
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load Model
try:
    model_path = os.path.join(os.getcwd(), "best_model.keras")
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

scaler_path = os.path.join(os.getcwd(), "scaler.pkl")
scaler = joblib.load(scaler_path)

load_dotenv()
apikey = os.getenv("API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

def fetch_last_24_hours_weather():
    base_url = "https://api.weatherapi.com/v1/history.json"
    latitude, longitude = 33.88, 35.48
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        "key": apikey,
        "q": f"{latitude},{longitude}",
        "dt": yesterday,
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()

        weather_data = []
        for hour in data["forecast"]["forecastday"][0]["hour"]:
            weather_data.append({
                "dt": hour["time"],
                "temp": hour["temp_c"],
                "pressure": hour["pressure_mb"],
                "humidity": hour["humidity"],
                "clouds": hour["cloud"],
                "wind_speed": hour["wind_kph"], 
                "wind_deg": hour["wind_degree"]
            })

        return {"status": "success", "data": weather_data}

    else:
        return {"status": "error", "message": f"Failed to fetch data. Status code: {response.status_code}"}


@app.get("/predict")
def predict_weather():
    try:
        weather_data = fetch_last_24_hours_weather()
        if weather_data["status"] == "error":
            return {"status": "error"}

        X_input = np.array([[d["temp"], d["pressure"], d["humidity"], d["clouds"], d["wind_speed"], d["wind_deg"]] for d in weather_data["data"]])
        X_input = scaler.transform(X_input).reshape(1, 24, 6)
        
        for i in range(24):
            today_pred = model.predict(X_input)
            X_input = X_input[:, 1:, :]
            today_pred = today_pred.reshape(1, 1, 6)
            X_input = np.append(X_input, today_pred, axis=1) 
    
        today_pred_inv = scaler.inverse_transform(X_input[0])

        predictions = [
            {
                "hour": i,
                "temperature": round(today_pred_inv[i][0])
            } for i in range(len(today_pred_inv))
        ]

        return {"status": "success", "predictions": predictions}

    except Exception as e:
        return {"status": "error", "message": str(e)}
