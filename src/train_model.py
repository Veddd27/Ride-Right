from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

def train_models(X_train, y_train):
    models = {}

    # Linear
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr

    # Ridge
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    models['Ridge'] = ridge

    # Lasso
    lasso = Lasso()
    lasso.fit(X_train, y_train)
    models['Lasso'] = lasso

    # Decision Tree
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb

    # LightGBM
    lgbm = lgb.LGBMRegressor()
    lgbm.fit(X_train, y_train)
    models['LightGBM'] = lgbm

    return models