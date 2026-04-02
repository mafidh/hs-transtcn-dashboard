import sys
import os

sys.path.append(os.path.abspath("."))

import streamlit as st
import torch
import matplotlib.pyplot as plt

from models.hybrid_model import HSTransTCN
from data.loaders import TimeSeriesDataset


# Page setup
st.set_page_config(
    page_title="HS-TransTCN Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.markdown("<h1 style='text-align: center;'>📊 HS-TransTCN Forecast Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load dataset
dataset = TimeSeriesDataset(
    "data/raw/airline-passengers.csv",
    input_window=12,
    forecast_horizon=3
)

# Sidebar
st.sidebar.header("⚙️ Controls")

index = st.sidebar.slider(
    "Select Time Window",
    0,
    len(dataset)-1,
    0
)

st.sidebar.markdown("---")
st.sidebar.info("Model: HS-TransTCN\n\nHybrid TCN + Transformer")

# Load model
model = HSTransTCN()
model.load_state_dict(torch.load("checkpoints/hs_transtcn_model.pth"))
model.eval()

# Get data
x, y, dates = dataset[index]

x_input = x.unsqueeze(0)

prediction = model(x_input).detach().numpy().flatten()
actual = y.numpy()

# Layout
col1, col2 = st.columns(2)

# 📈 Metrics Panel
with col1:
    st.subheader("📈 Prediction Summary")

    st.metric("Actual Avg", round(actual.mean(), 2))
    st.metric("Predicted Avg", round(prediction.mean(), 2))

    st.write("### Actual Values")
    st.write(actual)

    st.write("### Predicted Values")
    st.write(prediction)

    st.write("### Dates")
    st.write([str(d.date()) for d in dates])

# 📊 Graph Panel
with col2:
    st.subheader("📊 Forecast Visualization")

    fig, ax = plt.subplots()

    date_labels = [str(d.date()) for d in dates]

    ax.plot(date_labels, actual, marker='o', label="Actual")
    ax.plot(date_labels, prediction, marker='o', linestyle='--', label="Predicted")

    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    plt.xticks(rotation=45)

    st.pyplot(fig)

st.markdown("---")

st.markdown(
    "<center>🚀 HS-TransTCN | Final Year Project Dashboard</center>",
    unsafe_allow_html=True
)