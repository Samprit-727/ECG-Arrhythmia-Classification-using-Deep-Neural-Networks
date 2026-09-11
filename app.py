"""Interactive Streamlit demo for the ECG heartbeat classifiers."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unclassifiable"]
OUTPUT_DIR = Path("outputs")
MODEL_PATHS = {
    "Random Forest": OUTPUT_DIR / "random_forest_model.joblib",
    "CNN": OUTPUT_DIR / "cnn_model.keras",
    "SVM": OUTPUT_DIR / "svm_model.joblib",
    "ANN": OUTPUT_DIR / "ann_model.keras",
    "Decision Tree": OUTPUT_DIR / "decision_tree_model.joblib",
}

st.set_page_config(page_title="ECG Heartbeat Classifier", page_icon="❤️", layout="wide")


@st.cache_resource
def load_artifact(model_name: str):
    path = MODEL_PATHS[model_name]
    return tf.keras.models.load_model(path) if path.suffix == ".keras" else joblib.load(path)


def prepare_signals(uploaded_file) -> np.ndarray:
    """Accept 187 signal columns, optionally followed by one label column."""
    frame = pd.read_csv(uploaded_file, header=None)
    if frame.shape[1] == 188:
        frame = frame.iloc[:, :-1]
    if frame.shape[1] != 187:
        raise ValueError("The CSV must contain 187 signal columns, optionally followed by one label column.")
    return frame.to_numpy(dtype=np.float32)


st.title("ECG Heartbeat Classification")
st.caption("Interactive demo of SVM, Random Forest, Decision Tree, ANN, and 1D CNN models.")
st.warning("Educational demonstration only — this application is not a medical device and must not be used for diagnosis.")

with st.sidebar:
    available_models = [name for name, path in MODEL_PATHS.items() if path.exists()]
    if not available_models:
        st.error("No trained model files were found. Run `train_ecg_models.py` first.")
        st.stop()
    model_name = st.selectbox("Choose a model", available_models)
    unavailable_models = set(MODEL_PATHS) - set(available_models)
    if unavailable_models:
        st.caption("Run training again to add: " + ", ".join(sorted(unavailable_models)))
    uploaded_file = st.file_uploader("Upload an ECG CSV", type="csv")
    st.caption("Each row must have 187 ECG values. A 188th label column is allowed and will be ignored.")

comparison_path = OUTPUT_DIR / "model_comparison.csv"
if comparison_path.exists():
    st.subheader("Measured model comparison")
    st.dataframe(pd.read_csv(comparison_path), use_container_width=True, hide_index=True)

if uploaded_file is None:
    st.info("Upload a CSV file to classify an ECG beat. You may use `dataset hb/mitbih_test.csv` for a demo.")
    st.stop()

try:
    signals = prepare_signals(uploaded_file)
except (ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()

row_index = st.slider("Choose ECG row", 0, len(signals) - 1, 0)
signal = signals[row_index]

left, right = st.columns([3, 2])
with left:
    figure, axis = plt.subplots(figsize=(10, 3.5))
    axis.plot(signal, color="#e63946", linewidth=1.6)
    axis.set(title=f"ECG signal — row {row_index}", xlabel="Time step", ylabel="Amplitude")
    axis.grid(alpha=0.25)
    st.pyplot(figure, clear_figure=True)

with right:
    scaler_path = OUTPUT_DIR / "scaler.joblib"
    if not scaler_path.exists():
        st.error("Saved preprocessing artifacts are missing. Run `train_ecg_models.py` once after updating the project.")
        st.stop()
    model = load_artifact(model_name)
    if model is None:
        st.error(f"The saved {model_name} model is missing. Run the training script again to create it.")
        st.stop()

    scaled_signal = joblib.load(scaler_path).transform(signal.reshape(1, -1))
    if model_name == "CNN":
        probabilities = model.predict(scaled_signal[..., np.newaxis], verbose=0)[0]
    elif model_name == "ANN":
        probabilities = model.predict(scaled_signal, verbose=0)[0]
    elif hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_signal)[0]
    else:
        # SVM decision scores are converted to display-only relative probabilities.
        scores = model.decision_function(scaled_signal)[0]
        shifted_scores = scores - np.max(scores)
        probabilities = np.exp(shifted_scores) / np.exp(shifted_scores).sum()

    predicted_class = int(np.argmax(probabilities))
    st.subheader("Prediction")
    st.metric("Predicted heartbeat", CLASS_NAMES[predicted_class])
    st.metric("Confidence", f"{probabilities[predicted_class] * 100:.2f}%")
    probability_frame = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probabilities})
    st.bar_chart(probability_frame.set_index("Class"))
