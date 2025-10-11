#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App: Control Valve Characteristics (Installed Performance)
- Interactive visualization of linear vs. equal-percentage trims
- Adapted from original ipywidgets version
- Compatible with: streamlit run app_valve_characteristics.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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

def recompute():
    """Callback triggered when a slider changes."""
    st.session_state.recompute_flag = True

# -------------------------------------------------------------------------
# Initialize session state
# -------------------------------------------------------------------------
if "Cv" not in st.session_state:
    st.session_state.Cv = 5.0
if "DPt" not in st.session_state:
    st.session_state.DPt = 10.0
if "R" not in st.session_state:
    st.session_state.R = R_default
if "recompute_flag" not in st.session_state:
    st.session_state.recompute_flag = True

# -------------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------------
st.sidebar.header("Valve Settings")

st.sidebar.slider(
    "Valve Coefficient (Cv)",
    min_value=0.1,
    max_value=10.0,
    value=st.session_state.Cv,
    step=0.1,
    key="Cv",
    on_change=recompute
)

st.sidebar.slider(
    "Equal-Percentage R Factor",
    min_value=10.0,
    max_value=50.0,
    value=st.session_state.R,
    step=1.0,
    key="R",
    on_change=recompute
)

st.sidebar.slider(
    "Total Pressure Drop (ΔPt)",
    min_value=1.0,
    max_value=100,
    value=st.session_state.DPt,
    step=0.5,
    key="DPt",
    on_change=recompute
)

# -------------------------------------------------------------------------
# Compute and plot
# -------------------------------------------------------------------------
if st.session_state.recompute_flag:
    st.session_state.recompute_flag = False

    Cv = st.session_state.Cv
    DPt = st.session_state.DPt
    R = st.session_state.R

    lift = np.linspace(0, 1, 100)
    flow_lin = qi(lift, f_lin, Cv, R, DPt)
    flow_ep = qi(lift, f_ep, Cv, R, DPt)

    # Store data for download
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
    fig1, ax1 = plt.subplots(figsize=(7, 3))
    ax1.plot(lift, flow_lin, 'b-', label='Linear Valve')
    ax1.plot(lift, flow_ep, 'r--', label='Equal Percentage Valve')
    ax1.set_xlabel('Lift')
    ax1.set_ylabel('Flow')
    ax1.set_title('Flow vs. Lift')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig1)

    # -----------------------------------------------------------------
    # Plot 2 & 3: Pressure Drops (side-by-side)
    # -----------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(lift, DPt - DPe(flow_lin), 'k:', linewidth=2, label='Valve ΔP (Linear)')
        ax2.plot(lift, DPe(flow_lin), 'r--', linewidth=2, label='Equipment ΔP')
        ax2.set_title('Pressure Drops – Linear Valve', fontsize=11)
        ax2.set_xlabel('Lift')
        ax2.set_ylabel('ΔP')
        ax2.legend(fontsize=8)
        ax2.grid(True)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig2)

    with col2:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.plot(lift, DPt - DPe(flow_ep), 'k:', linewidth=2, label='Valve ΔP (Equal %)')
        ax3.plot(lift, DPe(flow_ep), 'r--', linewidth=2, label='Equipment ΔP')
        ax3.set_title('Pressure Drops – Equal-Percentage Valve', fontsize=11)
        ax3.set_xlabel('Lift')
        ax3.set_ylabel('ΔP')
        ax3.legend(fontsize=8)
        ax3.grid(True)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig3)

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
