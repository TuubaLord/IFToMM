from loaders import load_event, load_events
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from helpers import get_model



X_train, y_train = load_events([79, 46])
X_test, y_test = load_events([30, 65], noise=False)

model = get_model()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 1. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 2. Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# 3. R-squared (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)

print("--- XGBoost Regressor Evaluation ---")
print(f"Test RMSE: {rmse:.4f} (Lower is better)")
print(f"Test MAE:  {mae:.4f} (Lower is better)")
print(f"Test R² Score: {r2:.4f} (Closer to 1.0 is better)")
