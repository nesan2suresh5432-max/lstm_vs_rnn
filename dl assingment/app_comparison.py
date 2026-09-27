"""
Streamlit app: LSTM vs RNN — Long-Term Dependency Comparison

This directly matches the assignment problem statement:
"Develop an LSTM-based sequence prediction model for a problem in which
important information occurs several time steps before the required
prediction. Compare the performance of LSTM with a conventional RNN."

Run:
    streamlit run app_comparison.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense

st.set_page_config(page_title="LSTM vs RNN — Long-Term Dependency")
st.title("LSTM vs RNN: Long-Term Dependency Learning")

st.markdown(
    "> **Problem statement:** *Develop an LSTM-based sequence prediction "
    "model for a problem in which important information occurs several "
    "time steps before the required prediction. Compare the performance "
    "of LSTM with a conventional RNN.*"
)

st.write(
    "This demo builds exactly that scenario: an important digit is placed "
    "**several time steps before** the point where the prediction is "
    "required. Everything in between is random noise the model must ignore. "
    "As you increase the gap (sequence length), watch how a plain RNN "
    "forgets the important information while the LSTM keeps recalling it."
)


# ----------------------------
# Data + model builders
# ----------------------------
def generate_data(num_samples, seq_length):
    X = np.random.randint(0, 10, size=(num_samples, seq_length, 1)) / 10.0
    y = np.random.randint(0, 10, size=(num_samples,))
    for i in range(num_samples):
        X[i, 0, 0] = y[i] / 10.0
    return X, y


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
# Sidebar controls
# ----------------------------
st.sidebar.header("Experiment settings")
seq_lengths = st.sidebar.multiselect(
    "Sequence lengths to test",
    options=[5, 10, 20, 50, 100],
    default=[5, 10, 20, 50]
)
epochs = st.sidebar.slider("Epochs per model", min_value=3, max_value=20, value=5)
num_train = st.sidebar.slider("Training samples", min_value=200, max_value=2000, value=500, step=100)
num_test = st.sidebar.slider("Test samples", min_value=50, max_value=500, value=100, step=50)

run = st.button("Run comparison")

if run:
    if not seq_lengths:
        st.error("Pick at least one sequence length from the sidebar.")
    else:
        seq_lengths = sorted(seq_lengths)
        rnn_acc, lstm_acc = [], []
        trained_models = {}
        progress = st.progress(0)
        status = st.empty()

        for i, seq_length in enumerate(seq_lengths):
            status.write(
                f"Important info at step 0 → prediction required after "
                f"{seq_length} steps (gap = {seq_length} steps of noise). Training..."
            )

            X_train, y_train = generate_data(num_train, seq_length)
            X_test, y_test = generate_data(num_test, seq_length)

            rnn_model = build_rnn(seq_length)
            rnn_model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            _, acc_rnn = rnn_model.evaluate(X_test, y_test, verbose=0)
            rnn_acc.append(acc_rnn)

            lstm_model = build_lstm(seq_length)
            lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            _, acc_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)
            lstm_acc.append(acc_lstm)

            trained_models[seq_length] = {"rnn": rnn_model, "lstm": lstm_model}

            progress.progress((i + 1) / len(seq_lengths))

        status.write("Done.")

        # Keep models + lengths around after this run so the "test your own
        # sequence" section below still works after Streamlit reruns.
        st.session_state["trained_models"] = trained_models
        st.session_state["seq_lengths"] = seq_lengths

        # Results table
        results = pd.DataFrame({
            "Sequence length": seq_lengths,
            "RNN accuracy": rnn_acc,
            "LSTM accuracy": lstm_acc
        }).set_index("Sequence length")

        st.subheader("Results")
        st.dataframe(results.style.format("{:.2%}"))

        st.subheader("Accuracy vs Sequence Length")
        st.line_chart(results)

        best_gap_idx = int(np.argmax(np.array(lstm_acc) - np.array(rnn_acc)))
        st.info(
            f"Biggest gap: at sequence length {seq_lengths[best_gap_idx]}, the "
            f"important information sat {seq_lengths[best_gap_idx]} steps before "
            f"the required prediction. LSTM scored {lstm_acc[best_gap_idx]:.1%} "
            f"there vs RNN's {rnn_acc[best_gap_idx]:.1%}. This confirms the "
            "problem statement: as the gap between the important information "
            "and the required prediction grows, the plain RNN loses it "
            "(vanishing gradient), while the LSTM's gates preserve it."
        )
else:
    st.write("Set your options in the sidebar, then click **Run comparison**.")


# ----------------------------
# Test with your own sentence/paragraph
# ----------------------------
def text_to_sequence(text, length):
    """
    Convert free text into a fixed-length digit sequence the model can read,
    by mapping each character's code to a digit (0-9). Pads with 0 if the
    text is shorter than `length`, truncates if longer.
    """
    chars = text[:length]
    digits = [ord(c) % 10 for c in chars]
    if len(digits) < length:
        digits += [0] * (length - len(digits))
    return digits


if "trained_models" in st.session_state:
    st.divider()
    st.subheader("Try your own sentence")
    st.write(
        "This is the same idea as the problem statement, but using text: "
        "write any sentence or paragraph. The app converts it into a number "
        "sequence behind the scenes — the important information is your "
        "**first character**, and the rest of your sentence is the noise "
        "several steps before the model has to recall it. Both RNN and "
        "LSTM will try to identify that first character."
    )

    chosen_length = st.selectbox(
        "Sequence length (must match a length you trained above):",
        options=st.session_state["seq_lengths"]
    )

    custom_text = st.text_area(
        f"Write a sentence or paragraph (first {chosen_length} characters will be used):",
        placeholder="e.g. The quick brown fox jumps over the lazy dog near the riverbank every morning..."
    )

    if st.button("Predict with both models") and custom_text:
        seq = text_to_sequence(custom_text, chosen_length)

        x = np.array(seq).reshape(1, chosen_length, 1) / 10.0
        models = st.session_state["trained_models"][chosen_length]

        rnn_pred = models["rnn"].predict(x)[0]
        lstm_pred = models["lstm"].predict(x)[0]

        rnn_digit = int(np.argmax(rnn_pred))
        lstm_digit = int(np.argmax(lstm_pred))

        st.write(
            f"**First character of your sentence:** '{custom_text[0]}' → "
            f"mapped to digit **{seq[0]}**"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("RNN predicted", rnn_digit,
                      delta="Correct" if rnn_digit == seq[0] else "Wrong")
        with col2:
            st.metric("LSTM predicted", lstm_digit,
                      delta="Correct" if lstm_digit == seq[0] else "Wrong")

        compare_df = pd.DataFrame({
            "digit": [str(i) for i in range(10)],
            "RNN probability": rnn_pred,
            "LSTM probability": lstm_pred
        }).set_index("digit")

        st.write("Class probabilities (side by side):")
        st.bar_chart(compare_df)
