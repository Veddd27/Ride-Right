import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    return results