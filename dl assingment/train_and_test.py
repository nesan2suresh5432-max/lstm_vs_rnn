"""
Train a single final LSTM model on the delayed-recall task and save it
for deployment (used by app.py).
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

SEQ_LENGTH = 50  # fixed length the deployed app will expect


def generate_data(num_samples, seq_length):
    X = np.random.randint(0, 10, size=(num_samples, seq_length, 1)) / 10.0
    y = np.random.randint(0, 10, size=(num_samples,))
    for i in range(num_samples):
        X[i, 0, 0] = y[i] / 10.0
    return X, y


def build_lstm(seq_length):
    model = Sequential([
        LSTM(64, input_shape=(seq_length, 1)),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


if __name__ == "__main__":
    X_train, y_train = generate_data(2000, SEQ_LENGTH)
    X_test, y_test = generate_data(400, SEQ_LENGTH)

    model = build_lstm(SEQ_LENGTH)
    model.fit(X_train, y_train, epochs=10, batch_size=32,
              validation_data=(X_test, y_test))

    model.save("lstm_final.keras")
    print("Saved model as lstm_final.keras")