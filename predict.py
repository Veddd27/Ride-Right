import pickle
import numpy as np
import math
from datetime import datetime

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

print("=== Ride-Right Fare Predictor ===")

# User inputs coordinates
pickup_lat  = float(input("Pickup Latitude: "))
pickup_lon  = float(input("Pickup Longitude: "))
dropoff_lat = float(input("Dropoff Latitude: "))
dropoff_lon = float(input("Dropoff Longitude: "))
passengers  = int(input("Passenger Count: "))

# Auto compute everything else
distance = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
now = datetime.now()
hour       = now.hour
day_of_week = now.weekday()
month      = now.month
is_peak    = 1 if hour in range(7,10) or hour in range(17,20) else 0

# Build input — same order as X in main.py
sample = np.array([[pickup_lat, pickup_lon,
                    dropoff_lat, dropoff_lon,
                    distance, hour, day_of_week, month, passengers, is_peak]])

prediction = model.predict(sample)
print(f"\nPredicted Fare: ${prediction[0]:.2f}")