import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Band
from bokeh.models import Band, Legend
from bokeh.palettes import Category10
from bokeh.layouts import column

# --- ALS Baseline Correction Function ---
def baseline_als(data_y, lam, p, niter=10):
    L = len(data_y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)

    for _ in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D @ D.T
        z = spsolve(Z, w * data_y)
        w = p * (data_y > z) + (1 - p) * (data_y < z)

    return z

# --- Streamlit UI ---
st.title("FTIR Spectra Baseline Correction & Area Calculation")

# --- Sidebar for Inputs ---
st.sidebar.title("FTIR Spectra Settings")

# File uploader
uploaded_files = st.sidebar.file_uploader("Upload FTIR spectra of calibrants and analyte (columns: wavenumber, absorbance)", type="csv", accept_multiple_files=True)


# --- Smoothness (λ) Slider ---
use_large_range = st.sidebar.checkbox("Use larger smoothness range (1,000,000 - 10,000,000,000)")

if use_large_range:
    lam = st.sidebar.slider("Smoothness (λ)", min_value=100000, max_value=10000000000, value=10000000, step=100000)
else:
    lam = st.sidebar.slider("Smoothness (λ)", min_value=1000, max_value=100000, value=10000, step=1000)

lam_value = lam * 1e3  # Intern als Exponentialwert (z.B. 100 wird zu 1e5)
st.sidebar.write(f"Lamda (Smoothness): {lam_value:.0e}")  # Zeigt den berechneten Wert an

# --- Asymmetry (p) Slider ---
p = st.sidebar.slider("Asymmetry (p)", min_value=0.0001, max_value=0.01, value=0.001, step=0.00005)
st.sidebar.write(f"p (Asymmetry): {p:.3f}")  # Zeigt den ausgewählten p-Wert an


# Wavenumber selection
wn_min = st.sidebar.number_input("Wavenumber Min for integration", value=1000)
wn_max = st.sidebar.number_input("Wavenumber Max for integration", value=1800)

if uploaded_files:
    area_results = []
    plots = []

    palette = Category10[10]
    for idx, file in enumerate(uploaded_files):
        df = pd.read_csv(file, skiprows=1)
        df = df.sort_values(by=df.columns[0], ascending=False)  # Usually FTIR goes high->low wavenumber

        wn = df.iloc[:, 0].values
        absorbance = df.iloc[:, 1].values
        baseline = baseline_als(absorbance, lam, p)
        corrected = absorbance - baseline

        # Select range
        mask = (wn >= wn_min) & (wn <= wn_max)
        wn_sel = wn[mask]
        corr_sel = corrected[mask]
        base_sel = baseline[mask]

        # Calculate area under corrected spectrum
        area = abs(np.trapz(np.clip(corrected[mask], a_min=0, a_max=None), wn_sel))
        area_results.append((file.name, area))

        # --- Bokeh plot ---
        fig = figure(title=file.name[:-4], width=700, height=300, x_axis_label="Wavenumber", y_axis_label="Absorbance")
        fig.line(wn, absorbance, line_width=2, color="#CCCCCC", legend_label=file.name+' original')
        fig.line(wn, baseline, line_dash='dotted', color='gray', legend_label="Baseline")
        fig.line(wn, corrected, line_width=2, color=palette[idx % 10], legend_label=file.name+' corrected')

        # Create data source for Bokeh
        source = ColumnDataSource(data={
         'x': wn_sel,
         'lower': np.zeros_like(wn_sel),
         'upper': corr_sel
        })

        # Fill area under curve
        band = Band(base='x', lower='lower', upper='upper', source=source, fill_alpha=0.3, level='underlay', line_width=0, fill_color=palette[idx % 10])
        fig.add_layout(band)

        plots.append(fig)

    # Display plots
    st.bokeh_chart(column(*plots), use_container_width=True)

    # Display area results
    st.subheader("Integrated Areas (Corrected)")
    for name, area in area_results:
        st.write(f"**{name}**: {area:.4f} absorbance·cm⁻¹")