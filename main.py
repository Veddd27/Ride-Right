import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split

from src.data_preprocessing import clean_data
from src.feature_engineering import create_features
from src.train_model import train_models
from src.evaluate import evaluate


print("STARTING MODEL PIPELINE...")

# Step 1: Load data
df = pd.read_csv("data/uber.csv")

# Step 2: Clean data
print("Cleaning data...")
df = clean_data(df)

# Step 3: Feature Engineering
print("Creating features...")
df = create_features(df)

# Step 4: Select features and target
# NEW — coordinates retained as features
X = df[['pickup_latitude', 'pickup_longitude', 
        'dropoff_latitude', 'dropoff_longitude',
        'distance_km', 'hour', 'day_of_week', 'month', 'passenger_count', 'is_peak']]
y = df['fare_amount']

# Step 5: Train-Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train models
print("Training models...")
models = train_models(X_train, y_train)

# Step 7: Evaluate models
print("Evaluating models...")
results = evaluate(models, X_test, y_test)

# Print results
for model_name, metrics in results.items():
    print(f"\n{model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Step 8: Save best model (Random Forest)
print("\nSaving best model...")

os.makedirs("models", exist_ok=True)

best_model = models['Random Forest']

with open("models/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved successfully!")