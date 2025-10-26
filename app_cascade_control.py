"""
Streamlit App: PI vs. Cascade Control
- Interactive visualization comparing standard PI control and cascade control
- Uses only Streamlit charts (st.line_chart)
- Compatible with: streamlit run app_cascade_control.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint

# -------------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------------
st.set_page_config(page_title="PI vs. Cascade Control", layout="wide")
st.title("PI vs. Cascade Control Simulation")
st.caption("Compare a single-loop PI controller with a two-loop cascade control structure.")

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
n = 1201
tf = 1200.0
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
    """Standard PI control."""
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
    """Cascade control simulation."""
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
# Sidebar controls
# -------------------------------------------------------------------------
st.sidebar.header("Controller Settings")

mode = st.sidebar.radio(
    "Control Type",
    ["PI Control", "Cascade Control"],
    help="Choose between standard PI and two-loop cascade control."
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
# Run simulation
# -------------------------------------------------------------------------
if mode == "PI Control":
    t, PV1, PV2, SP2, OP, iae = simulate_PI(Kc, tauI)
    df = pd.DataFrame({
        "Time (s)": t,
        "Temperature 1": PV1,
        "Temperature 2": PV2,
        "Setpoint": SP2,
        "Heater Output": OP
    })
else:
    t, PV1, PV2, SP1, SP2, OP, iae = simulate_cascade(Kc1, tauI1, Kc2, tauI2)
    df = pd.DataFrame({
        "Time (s)": t,
        "Temperature 1": PV1,
        "Temperature 2": PV2,
        "Setpoint 1": SP1,
        "Setpoint 2": SP2,
        "Heater Output": OP
    })

# -------------------------------------------------------------------------
# Streamlit line charts
# -------------------------------------------------------------------------
st.subheader("Temperature Response")

if mode == "PI Control":
    temp_df = df[["Time (s)", "Temperature 1", "Temperature 2", "Setpoint"]].set_index("Time (s)")
else:
    temp_df = df[["Time (s)", "Temperature 1", "Temperature 2", "Setpoint 1", "Setpoint 2"]].set_index("Time (s)")

st.line_chart(temp_df, x_label="Time (s)", y_label="Temperature (°C)")

st.subheader("Heater Output (%)")
op_df = df[["Time (s)", "Heater Output"]].set_index("Time (s)")
st.line_chart(op_df, x_label="Time (s)", y_label="Heater (%)")

# -------------------------------------------------------------------------
# Results summary
# -------------------------------------------------------------------------
if mode == "PI Control":
    st.success(f"PI Control → Kc={Kc:.2f}, τI={tauI:.1f}, IAE={iae:.1f}")
else:
    st.success(f"Cascade Control → Kc1={Kc1:.2f}, τI1={tauI1:.1f}, Kc2={Kc2:.2f}, τI2={tauI2:.1f}, IAE={iae:.1f}")

# -------------------------------------------------------------------------
# Download data
# -------------------------------------------------------------------------
st.download_button(
    label="Download Simulation Data as CSV",
    data=df.to_csv(index=False),
    file_name=f"{mode.replace(' ', '_').lower()}_data.csv",
    mime="text/csv"
)
