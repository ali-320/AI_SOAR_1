import numpy as np
from tensorflow.keras.models import load_model

def load_models(fnn_path, cnn_path, lstm_path):
    """Load the trained FNN, CNN, and LSTM models."""
    fnn = load_model(fnn_path)
    cnn = load_model(cnn_path)
    lstm = load_model(lstm_path)
    return fnn, cnn, lstm

def predict_ensemble(fnn, cnn, lstm, X_fnn, X_cnn, X_lstm):
    """Make predictions using all three models and average them."""
    pred_fnn = fnn.predict(X_fnn)
    pred_cnn = cnn.predict(X_cnn)
    pred_lstm = lstm.predict(X_lstm)
    ensemble_pred = (pred_fnn + pred_cnn + pred_lstm) / 3
    return ensemble_pred

if __name__ == "__main__":
    # Load models
    fnn, cnn, lstm = load_models("./Preprocessing/DL_Models/fnn_model.h5", "./Preprocessing/DL_Models/cnn_model.h5", "./Preprocessing/DL_Models/lstm_model.h5")

    # Load test data
    X_fnn_test = np.load("./Preprocessing/DL_Data/FNN/fnn_test_X.npy")
    X_cnn_test = np.load("./Preprocessing/DL_Data/CNN/cnn_test_X.npy")
    X_lstm_test = np.load("./Preprocessing/DL_Data/LSTM/lstm_test_X.npy")
    y_test = np.load("./Preprocessing/DL_Data/FNN/fnn_test_y.npy")  # Same labels for all

    # Make ensemble predictions
    ensemble_pred = predict_ensemble(fnn, cnn, lstm, X_fnn_test, X_cnn_test, X_lstm_test)
    accuracy = np.mean((ensemble_pred > 0.5) == y_test)
    print(f"Ensemble test accuracy: {accuracy}")