import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))

    return R * c


def create_features(df):
    # Keep computing distance
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                                   df['dropoff_latitude'], df['dropoff_longitude'])
    
    # Extract time features
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['is_peak'] = df['hour'].apply(lambda x: 1 if x in range(7,10) or x in range(17,20) else 0)
    
    return df