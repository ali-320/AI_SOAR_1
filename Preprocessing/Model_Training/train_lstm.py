import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------
# 1) Load data via memmap
# -------------------------
X_train = np.load("./Preprocessing/DL_Data/LSTM/lstm_train_X.npy", mmap_mode="r")
y_train = np.load("./Preprocessing/DL_Data/LSTM/lstm_train_y.npy", mmap_mode="r")

X_val   = np.load("./Preprocessing/DL_Data/LSTM/lstm_val_X.npy",   mmap_mode="r")
y_val   = np.load("./Preprocessing/DL_Data/LSTM/lstm_val_y.npy",   mmap_mode="r")

X_test  = np.load("./Preprocessing/DL_Data/LSTM/lstm_test_X.npy",  mmap_mode="r")
y_test  = np.load("./Preprocessing/DL_Data/LSTM/lstm_test_y.npy",  mmap_mode="r")

batch_size = 32

def make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=16384, reshuffle_each_iteration=True)
    # cast on the fly to avoid loading everything into RAM as float64
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), tf.cast(b, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(X_train, y_train, shuffle=True)
val_ds   = make_ds(X_val,   y_val)
test_ds  = make_ds(X_test,  y_test)

# -------------------------
# 2) Build the LSTM model
# -------------------------
timesteps = X_train.shape[1]
features  = X_train.shape[2]

model = Sequential([
    LSTM(64, input_shape=(timesteps, features), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------
# 3) Train (with safe callbacks)
# -------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("./Preprocessing/DL_Models/lstm_model.h5",
                    monitor="val_loss", save_best_only=True)
]

model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=callbacks)

# -------------------------
# 4) Test evaluation
# -------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

# (Optional) explicit save if you want the final weights too:
model.save("./Preprocessing/DL_Models/lstm_model_final.h5")
print("Training complete. Best checkpoint and final model saved.")
