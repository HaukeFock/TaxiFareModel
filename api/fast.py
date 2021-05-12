
# write some code for the API here

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello Hauke"}

@app.get("/predict_fare")
def predict_fare(key, pickup_datetime,pickup_longitude , pickup_latitude , dropoff_longitude , dropoff_latitude , passenger_count):
    array = {"key": 1,
    "pickup_datetime": pickup_datetime,
    "pickup_longitude": pickup_longitude,
    "pickup_latitude": pickup_latitude,
    "dropoff_longitude": dropoff_longitude,
    "dropoff_latitude": dropoff_latitude,
    "passenger_count": passenger_count}
    loaded_model = joblib.load('/home/hauke/code/HaukeFock/TaxiFareModel/TaxiFareModel/models_simpletaxifare_model.joblib')
    X_pred = pd.DataFrame({k: [v] for k, v in array.items()})
    X_pred.iloc[:,2:6] = X_pred.iloc[:,2:6].astype('float64')
    X_pred.iloc[:,6] = X_pred.iloc[:,6].astype('int64')
    print(X_pred)
    y_pred = loaded_model.predict(X_pred)
    print(y_pred)
    return array, {"prediction":str(y_pred[0])}

