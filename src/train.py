from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
from src.preprocess import preprocess_data
from src.config import MODEL_PATH, SCALER_PATH

def train_model():
    X_train, X_test, y_train, y_test, scaler = preprocess_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Model trained | MAE: {error:.2f}")
    return error
