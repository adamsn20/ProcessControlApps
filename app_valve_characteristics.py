
"""
Streamlit App: Control Valve Characteristics (Installed Performance)
- Interactive visualization of linear vs. equal-percentage trims
- Adapted from Dr. John Hedengren's original ipywidgets version
- Compatible with: streamlit run app_valve_characteristics.py
"""

import streamlit as st
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------------
st.set_page_config(page_title="Valve Performance Visualization", layout="wide")
st.title("Valve Performance – Installed Characteristics")
st.caption("Compare linear and equal-percentage valve trims under installed conditions.")

# -------------------------------------------------------------------------
# Constants and parameters
# -------------------------------------------------------------------------
c1 = 2.0     # Coefficient for pressure drop across equipment
g_s = 1.1    # Specific gravity of fluid
R_default = 30.0  # Typical equal-percentage valve characteristic factor

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def f_lin(x, R):
    """Linear valve trim."""
    return x

def f_ep(x, R):
    """Equal-percentage valve trim (R = 20–50 typical)."""
    return R**(x - 1)

def DPe(q):
    """Pressure drop across process equipment."""
    return c1 * q**2

def qi(x, f, Cv, R, DPt):
    """Flow rate through the valve and process system."""
    return np.sqrt((Cv * f(x, R))**2 * DPt / (g_s + (Cv * f(x, R))**2 * c1))

# -------------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------------
st.sidebar.header("Valve Settings")

Cv = st.sidebar.slider(
    "Valve Coefficient (Cv)",
    min_value=0.1,
    max_value=10.0,
    value=5.0,
    step=0.1,
)

R = st.sidebar.slider(
    "Equal-Percentage R Factor",
    min_value=10.0,
    max_value=50.0,
    value=30.0,
    step=1.0,
)

DPt = st.sidebar.slider(
    "Total Pressure Drop (ΔPt)",
    min_value=1.0,
    max_value=100.0,
    value=100.0,
    step=0.5,
)

show_desired_profile = st.sidebar.checkbox(
    "Show Desired Profile",
    value=False,
    help="Add a black reference line to the Flow vs. Lift plot.",
)

# -------------------------------------------------------------------------
# Compute and plot
# -------------------------------------------------------------------------

lift = np.linspace(0, 1, 100)
flow_lin = qi(lift, f_lin, Cv, R, DPt)
flow_ep = qi(lift, f_ep, Cv, R, DPt)

data = {
    "Lift": lift,
    "Flow_Linear": flow_lin,
    "Flow_Equal%": flow_ep,
    "ValveDP_Linear": DPt - DPe(flow_lin),
    "ValveDP_Equal%": DPt - DPe(flow_ep),
    "EquipDP_Linear": DPe(flow_lin),
    "EquipDP_Equal%": DPe(flow_ep),
}

# -----------------------------------------------------------------
# Plot 1: Flow vs. Lift
# -----------------------------------------------------------------
flow_data = pd.DataFrame({
    "Lift": lift,
    "Linear Valve": flow_lin,
    "Equal Percentage Valve": flow_ep,
}).set_index("Lift")

desired_profile = pd.DataFrame({
    "Lift": lift,
    "Linear Valve": flow_lin,
    "Equal Percentage Valve": flow_ep,
    "Desired Profile": np.linspace(0, 9.4, 100)
}).set_index("Lift")

st.subheader("Flow vs. Lift")
if show_desired_profile:
    st.line_chart(desired_profile, color = ["#0000FF","#FF0000","#000000"], x_label = "Lift", y_label = "Flow")
else:
    st.line_chart(flow_data, color = ["#0000FF","#FF0000"], x_label = "Lift", y_label = "Flow")

# -----------------------------------------------------------------
# Plot 2 & 3: Pressure Drops (side-by-side)
# -----------------------------------------------------------------
st.subheader("Pressure Drop vs. Lift")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Linear Valve")
    dp_linear_data = pd.DataFrame({
        "Lift": lift,
        "Valve ΔP (Linear)": DPt - DPe(flow_lin),
        "Equipment ΔP": DPe(flow_lin),
    }).set_index("Lift")
    st.line_chart(dp_linear_data, color = ["#00FFFF","#0000FF"], x_label = "Lift", y_label = "ΔP")

with col2:
    st.subheader("Equal-Percentage Valve")
    dp_equal_data = pd.DataFrame({
        "Lift": lift,
        "Valve ΔP (Equal %)": DPt - DPe(flow_ep),
        "Equipment ΔP": DPe(flow_ep),
    }).set_index("Lift")
    st.line_chart(dp_equal_data, color = ["#00FFFF","#0000FF"], x_label = "Lift", y_label = "ΔP")

# -----------------------------------------------------------------
# Footer and Data Table
# -----------------------------------------------------------------
st.success(f"Simulation complete: Cv={Cv:.2f}, R={R:.1f}, ΔPt={DPt:.1f}")

df = pd.DataFrame(data)
st.download_button(
    "Download Data as CSV",
    df.to_csv(index=False),
    file_name="valve_performance.csv",
    mime="text/csv"
)
