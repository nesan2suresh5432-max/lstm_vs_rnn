"""
LSTM for Long-Term Dependency Learning
----------------------------------------
Task: Delayed recall problem.
A digit (0-9) appears at the very first time step of a sequence.
The rest of the sequence is random noise. The model must predict
that first digit after seeing the whole sequence.

We train both a SimpleRNN and an LSTM on this task across increasing
sequence lengths, and compare their accuracy to show how LSTM handles
long-term dependencies better than a plain RNN.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# ----------------------------
# 1. Data generation
# ----------------------------
def generate_data(num_samples, seq_length):
    """
    Create sequences where the label digit is placed at position 0,
    followed by random noise. Model must recall position-0 digit.
    """
    X = np.random.randint(0, 10, size=(num_samples, seq_length, 1)) / 10.0
    y = np.random.randint(0, 10, size=(num_samples,))
    for i in range(num_samples):
        X[i, 0, 0] = y[i] / 10.0
    return X, y


# ----------------------------
# 2. Model builders
# ----------------------------
def build_rnn(seq_length):
    model = Sequential([
        SimpleRNN(64, input_shape=(seq_length, 1)),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_lstm(seq_length):
    model = Sequential([
        LSTM(64, input_shape=(seq_length, 1)),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ----------------------------
# 3. Experiment across sequence lengths
# ----------------------------
def run_experiment(seq_lengths, num_train=2000, num_test=400, epochs=10):
    rnn_acc = []
    lstm_acc = []

    for seq_length in seq_lengths:
        print(f"\n=== Sequence length: {seq_length} ===")

        X_train, y_train = generate_data(num_train, seq_length)
        X_test, y_test = generate_data(num_test, seq_length)

        # --- RNN ---
        rnn_model = build_rnn(seq_length)
        rnn_model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        _, acc_rnn = rnn_model.evaluate(X_test, y_test, verbose=0)
        rnn_acc.append(acc_rnn)
        print(f"RNN  test accuracy: {acc_rnn:.3f}")

        # --- LSTM ---
        lstm_model = build_lstm(seq_length)
        lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        _, acc_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)
        lstm_acc.append(acc_lstm)
        print(f"LSTM test accuracy: {acc_lstm:.3f}")

    return rnn_acc, lstm_acc


# ----------------------------
# 4. Plot results
# ----------------------------
def plot_results(seq_lengths, rnn_acc, lstm_acc):
    plt.figure(figsize=(8, 5))
    plt.plot(seq_lengths, rnn_acc, marker="o", label="SimpleRNN")
    plt.plot(seq_lengths, lstm_acc, marker="o", label="LSTM")
    plt.axhline(0.1, color="gray", linestyle="--", label="Chance level (10%)")
    plt.xlabel("Sequence length (noise steps)")
    plt.ylabel("Test accuracy")
    plt.title("LSTM vs RNN: Long-Term Dependency Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lstm_vs_rnn_accuracy.png", dpi=150)
    print("\nPlot saved as lstm_vs_rnn_accuracy.png")
    plt.show()


# ----------------------------
# 5. Main
# ----------------------------
if __name__ == "__main__":
    seq_lengths = [5, 10, 20, 50, 100]
    rnn_acc, lstm_acc = run_experiment(seq_lengths, epochs=10)
    plot_results(seq_lengths, rnn_acc, lstm_acc)

    print("\nFinal comparison:")
    for s, r, l in zip(seq_lengths, rnn_acc, lstm_acc):
        print(f"seq_length={s:>4} | RNN acc={r:.3f} | LSTM acc={l:.3f}")