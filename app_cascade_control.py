"""
Streamlit App: PI vs. Cascade Control
- Interactive visualization comparing standard PI control and cascade control
- Adapted from Dr. John Hedengren's original ipywidgets version
- Compatible with: streamlit run app_cascade_control.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# -------------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------------
st.set_page_config(page_title="PI vs. Cascade Control", layout="wide")
st.title("PI vs. Cascade Control Simulation")
st.caption("Compare a single-loop PI controller to a two-loop cascade control structure.")

# -------------------------------------------------------------------------
# Constants and parameters
# -------------------------------------------------------------------------
n = 1201   # time points
tf = 1200.0  # total time
Kp = 0.8473
Kd = 0.3
taus = 51.08
zeta = 1.581
thetap = 0.0

# -------------------------------------------------------------------------
# Process model
# -------------------------------------------------------------------------
def process(z, t, u):
    x1, y1, x2, y2 = z
    dx1dt = (1.0 / (taus ** 2)) * (-2.0 * zeta * taus * x1 - (y1 - 23.0) + Kp * u + Kd * (y2 - y1))
    dy1dt = x1
    dx2dt = (1.0 / (taus ** 2)) * (-2.0 * zeta * taus * x2 - (y2 - 23.0) + Kd * (y1 - y2))
    dy2dt = x2
    return [dx1dt, dy1dt, dx2dt, dy2dt]

# -------------------------------------------------------------------------
# Simulation functions
# -------------------------------------------------------------------------
def simulate_PI(Kc, tauI):
    """Simulate standard PI control."""
    t = np.linspace(0, tf, n)
    P, I, e, ie = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    OP = np.zeros(n)
    PV1, PV2 = np.ones(n) * 23.0, np.ones(n) * 23.0
    SP2 = np.ones(n) * 23.0
    SP2[10:] = 35.0
    z0 = [0, 23.0, 0, 23.0]

    for i in range(1, n):
        ts = [t[i - 1], t[i]]
        z = odeint(process, z0, ts, args=(OP[max(0, i - 1 - int(thetap))],))
        z0 = z[1]
        PV1[i], PV2[i] = z0[1], z0[3]
        e[i] = SP2[i] - PV2[i]
        ie[i] = ie[i - 1] + e[i]
        dt = t[i] - t[i - 1]
        P[i] = Kc * e[i]
        I[i] = Kc / tauI * ie[i]
        OP[i] = np.clip(P[i] + I[i], 0, 100)
        if OP[i] in [0, 100]:
            ie[i] = ie[i - 1]  # anti-windup
    iae = np.sum(np.abs(e))
    return t, PV1, PV2, SP2, OP, iae


def simulate_cascade(Kc1, tauI1, Kc2, tauI2):
    """Simulate cascade control."""
    t = np.linspace(0, tf, n)
    P1, I1, e1, ie1 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    P2, I2, e2, ie2 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    OP, SP1 = np.zeros(n), np.ones(n) * 23.0
    PV1, PV2 = np.ones(n) * 23.0, np.ones(n) * 23.0
    SP2 = np.ones(n) * 23.0
    SP2[10:] = 35.0
    z0 = [0, 23.0, 0, 23.0]

    for i in range(1, n):
        ts = [t[i - 1], t[i]]
        z = odeint(process, z0, ts, args=(OP[max(0, i - 1 - int(thetap))],))
        z0 = z[1]
        PV1[i], PV2[i] = z0[1], z0[3]
        dt = t[i] - t[i - 1]

        # Outer loop
        e2[i] = SP2[i] - PV2[i]
        ie2[i] = ie2[i - 1] + e2[i] * dt
        P2[i] = Kc2 * e2[i]
        I2[i] = Kc2 / tauI2 * ie2[i]
        SP1[i] = np.clip(P2[i] + I2[i], 23, 85)
        if SP1[i] in [23, 85]:
            ie2[i] = ie2[i - 1]

        # Inner loop
        e1[i] = SP1[i] - PV1[i]
        ie1[i] = ie1[i - 1] + e1[i] * dt
        P1[i] = Kc1 * e1[i]
        I1[i] = Kc1 / tauI1 * ie1[i]
        OP[i] = np.clip(P1[i] + I1[i], 0, 100)
        if OP[i] in [0, 100]:
            ie1[i] = ie1[i - 1]

    iae = np.sum(np.abs(e2))
    return t, PV1, PV2, SP1, SP2, OP, iae

# -------------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------------
st.sidebar.header("Simulation Settings")

mode = st.sidebar.radio(
    "Select Control Type",
    ["PI Control", "Cascade Control"],
    help="Choose between standard PI and two-loop cascade structure."
)

if mode == "PI Control":
    Kc = st.sidebar.slider("Kc", 1.0, 10.0, 2.0, 0.5)
    tauI = st.sidebar.slider("τI", 5.0, 300.0, 150.0, 5.0)
else:
    Kc1 = st.sidebar.slider("Inner Loop Kc1", 1.0, 10.0, 2.0, 0.5)
    tauI1 = st.sidebar.slider("Inner Loop τI1", 5.0, 300.0, 200.0, 5.0)
    Kc2 = st.sidebar.slider("Outer Loop Kc2", 2.0, 10.0, 3.0, 0.5)
    tauI2 = st.sidebar.slider("Outer Loop τI2", 5.0, 300.0, 150.0, 5.0)

# -------------------------------------------------------------------------
# Run Simulation
# -------------------------------------------------------------------------
if mode == "PI Control":
    t, PV1, PV2, SP2, OP, iae = simulate_PI(Kc, tauI)
else:
    t, PV1, PV2, SP1, SP2, OP, iae = simulate_cascade(Kc1, tauI1, Kc2, tauI2)

# -------------------------------------------------------------------------
# Plot Results
# -------------------------------------------------------------------------
st.subheader("Temperature Responses")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(t, PV1, "r-", label="Temperature 1")
ax[0].plot(t, PV2, "b--", label="Temperature 2")
if mode == "Cascade Control":
    ax[0].plot(t, SP1, "k:", label="T1 Setpoint (SP1)")
ax[0].plot(t, SP2, "k-", label="T2 Setpoint (SP2)")
ax[0].set_xlabel("Time (sec)")
ax[0].set_ylabel("Temperature (°C)")
ax[0].legend(loc="best")
ax[0].grid(True)

ax[1].plot(t, OP, "r-", label="Heater Output (%)")
ax[1].set_xlabel("Time (sec)")
ax[1].set_ylabel("Heater (%)")
ax[1].legend(loc="best")
ax[1].grid(True)
st.pyplot(fig)

# -------------------------------------------------------------------------
# Performance Metrics
# -------------------------------------------------------------------------
if mode == "PI Control":
    st.success(f"PI Control Results → Kc={Kc:.2f}, τI={tauI:.1f}, IAE={iae:.1f}")
else:
    st.success(f"Cascade Control Results → Kc1={Kc1:.2f}, τI1={tauI1:.1f}, Kc2={Kc2:.2f}, τI2={tauI2:.1f}, IAE={iae:.1f}")

# -------------------------------------------------------------------------
# Data Download
# -------------------------------------------------------------------------
if mode == "PI Control":
    df = pd.DataFrame({
        "Time": t,
        "Temperature1": PV1,
        "Temperature2": PV2,
        "Setpoint": SP2,
        "Heater": OP
    })
else:
    df = pd.DataFrame({
        "Time": t,
        "Temperature1": PV1,
        "Temperature2": PV2,
        "Setpoint1": SP1,
        "Setpoint2": SP2,
        "Heater": OP
    })

st.download_button(
    "Download Simulation Data as CSV",
    df.to_csv(index=False),
    file_name=f"{mode.replace(' ', '_').lower()}_data.csv",
    mime="text/csv"
)
