import streamlit as st
import numpy as np
from src.simulate_fatigue import simulate_damage
from joblib import load

st.title("Digital Twin: Cantilever Beam Fatigue")

force = st.slider("Applied Force (N)", 100, 600, 300)
cycles = st.slider("Load Cycles", 1000, 20000, 5000)

model = load("trained_model.joblib")
pred_damage = model.predict([[force, cycles]])[0]

st.write(f"### Predicted Damage: {pred_damage:.4f}")
if pred_damage >= 1.0:
    st.error("Beam has failed.")
else:
    st.success("Beam is still operational.")

# Optional: add visual or deflection chart
