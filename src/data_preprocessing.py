import pandas as pd

def clean_data(df):
    # Drop unwanted column
    df = df.drop(columns=['key'], errors='ignore')

    # Remove nulls
    df = df.dropna()

    # Clean fare
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)]

    # Clean passenger count
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

    # Clean coordinates
    df = df[
        (df['pickup_latitude'].between(-90, 90)) &
        (df['pickup_longitude'].between(-180, 180)) &
        (df['dropoff_latitude'].between(-90, 90)) &
        (df['dropoff_longitude'].between(-180, 180))
    ]

    return df