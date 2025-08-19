import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_fnn(input_shape):
    """Build a simple FNN model."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification for 'evil'
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load training and validation data
    X_train = np.load("./Preprocessing/DL_Data/FNN/fnn_train_X.npy")
    y_train = np.load("./Preprocessing/DL_Data/FNN/fnn_train_y.npy")
    X_val = np.load("./Preprocessing/DL_Data/FNN/fnn_val_X.npy")
    y_val = np.load("./Preprocessing/DL_Data/FNN/fnn_val_y.npy")

    # Build and train model
    model = build_fnn(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate on test data
    X_test = np.load("./Preprocessing/DL_Data/FNN/fnn_test_X.npy")
    y_test = np.load("./Preprocessing/DL_Data/FNN/fnn_test_y.npy")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}")

    # Save model
    model.save("./Preprocessing/DL_Models/fnn_model.h5")
    print("FNN model trained and saved.")