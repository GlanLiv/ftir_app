import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LinearRegression
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Band
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
st.title("FTIR Spectra Baseline Correction & Calibration")

st.sidebar.title("FTIR Spectra Settings")

uploaded_files = st.sidebar.file_uploader("Calibrant spectra (columns: wavenumber, absorbance)", type="csv", accept_multiple_files=True)
uploaded_analyt = st.sidebar.file_uploader("Analyte spectra (columns: wavenumber, absorbance)", type="csv", accept_multiple_files=True)

use_large_range = st.sidebar.checkbox("Use larger smoothness range (1,000,000 - 10,000,000,000)")

if use_large_range:
    lam = st.sidebar.slider("Smoothness (λ)", min_value=100000, max_value=10000000000, value=10000000, step=100000)
else:
    lam = st.sidebar.slider("Smoothness (λ)", min_value=1000, max_value=100000, value=10000, step=1000)

lam_value = lam * 1e3
st.sidebar.write(f"Lambda (Smoothness): {lam_value:.0e}")

p = st.sidebar.slider("Asymmetry (p)", min_value=0.0001, max_value=0.01, value=0.001, step=0.00005)
st.sidebar.write(f"p (Asymmetry): {p:.3f}")

wn_min = st.sidebar.number_input("Wavenumber Min", value=1000)
wn_max = st.sidebar.number_input("Wavenumber Max", value=1800)

area_results = []
plots = []
palette = Category10[10]

# --- Process Calibrants ---
if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        df = pd.read_csv(file, skiprows=1)
        df = df.sort_values(by=df.columns[0], ascending=False)

        wn = df.iloc[:, 0].values
        absorbance = df.iloc[:, 1].values
        baseline = baseline_als(absorbance, lam, p)
        corrected = absorbance - baseline

        mask = (wn >= wn_min) & (wn <= wn_max)
        wn_sel = wn[mask]
        corr_sel = corrected[mask]
        area = abs(np.trapz(np.clip(corrected[mask], a_min=0, a_max=None), wn_sel))
        area_results.append((file.name, area))

        fig = figure(title=file.name, width=700, height=300, x_axis_label="Wavenumber", y_axis_label="Absorbance")
        fig.line(wn, absorbance, line_width=2, color="#CCCCCC", legend_label=file.name+' original')
        fig.line(wn, baseline, line_dash='dotted', color='gray', legend_label="Baseline")
        fig.line(wn, corrected, line_width=2, color=palette[idx % 10], legend_label=file.name+' corrected')

        source = ColumnDataSource(data={'x': wn_sel, 'lower': np.zeros_like(wn_sel), 'upper': corr_sel})
        band = Band(base='x', lower='lower', upper='upper', source=source, fill_alpha=0.3, level='underlay', line_width=0, fill_color=palette[idx % 10])
        fig.add_layout(band)

        plots.append(fig)

# --- Process Analytes ---
analyte_areas = []
analyte_concentrations = []
if uploaded_analyt:
    if not isinstance(uploaded_analyt, list):
        uploaded_analyt = [uploaded_analyt]

    for idx, file in enumerate(uploaded_analyt):
        df = pd.read_csv(file, skiprows=1)
        df = df.sort_values(by=df.columns[0], ascending=False)
        wn = df.iloc[:, 0].values
        absorbance = df.iloc[:, 1].values
        baseline = baseline_als(absorbance, lam, p)
        corrected = absorbance - baseline

        mask = (wn >= wn_min) & (wn <= wn_max)
        wn_sel = wn[mask]
        corr_sel = corrected[mask]
        area = abs(np.trapz(np.clip(corrected[mask], a_min=0, a_max=None), wn_sel))
        analyte_areas.append((file.name, area))

        fig = figure(title=f"Analyte: {file.name}", width=700, height=300, x_axis_label="Wavenumber", y_axis_label="Absorbance")
        fig.line(wn, absorbance, line_width=2, color="#CCCCCC", legend_label="Original")
        fig.line(wn, baseline, line_dash='dotted', color='gray', legend_label="Baseline")
        fig.line(wn, corrected, line_width=2, color='red', legend_label="Corrected")

        source = ColumnDataSource(data={'x': wn_sel, 'lower': np.zeros_like(wn_sel), 'upper': corr_sel})
        band = Band(base='x', lower='lower', upper='upper', source=source, fill_alpha=0.3, level='underlay', fill_color='red')
        fig.add_layout(band)

        plots.append(fig)

# --- Show All Spectra Plots ---
if plots:
    st.bokeh_chart(column(*plots), use_container_width=True)

# --- Display Areas ---
if area_results:
    st.subheader("Calibrant Areas")
    for name, area in area_results:
        st.write(f"**{name}**: {area:.4f} absorbance·cm⁻¹")

if analyte_areas:
    st.subheader("Analyte Areas")
    for name, area in analyte_areas:
        st.write(f"**{name}**: {area:.4f} absorbance·cm⁻¹")

# --- Calibration Section ---
if uploaded_analyt and len(area_results) >= 3:
    st.subheader("Calibration Input")
    concentrations = []
    with st.form("calibration_form"):
        for name, _ in area_results:
            concentration = st.number_input(f"Concentration for **{name}** (same units)", min_value=0.0, step=0.01, key=name)
            concentrations.append(concentration)
        submitted = st.form_submit_button("Run Calibration")

    if submitted:
        y = np.array([area for _, area in area_results]).reshape(-1, 1)
        x = np.array(concentrations).reshape(-1, 1)

        model = LinearRegression()
        model.fit(x, y)

        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        r2 = model.score(x, y)

        st.markdown(f"**Regression Equation:**  y = {intercept:.4f} + {slope:.4f}·x")
        st.markdown(f"**R²:** {r2:.4f}")

        st.subheader("Predicted Analyte Concentrations")
        for name, area in analyte_areas:
            conc = (area - intercept) / slope
            analyte_concentrations.append(conc)
            st.write(f"**{name}** → {conc:.4f} (same units)")

        conc_array = np.array(analyte_concentrations)
        mean_conc = conc_array.mean()
        std_conc = conc_array.std()

        st.markdown(f"### Final Result: {mean_conc:.4f} ± {std_conc:.4f} (mean ± std)")

        calib_plot = figure(title="Calibration Curve", x_axis_label="Concentration", y_axis_label="Integrated Area", width=700, height=400)
        calib_plot.circle(x.flatten(), y.flatten(), size=10, color="blue", legend_label="Calibrants")
        x_fit = np.linspace(min(x)[0], max(x)[0], 100).reshape(-1, 1)
        y_fit = model.predict(x_fit)
        calib_plot.line(x_fit.flatten(), y_fit.flatten(), color="black", line_width=2, legend_label=f"Fit: y = {intercept:.2f} + {slope:.2f}·x | R² = {r2:.4f}")

        for name, area in analyte_areas:
            conc = (area - intercept) / slope
            calib_plot.square([conc], [area], size=12, color="red", legend_label="Analyte")

        calib_plot.legend.location = "top_left"
        calib_plot.legend.label_text_font_size = "10pt"

        st.bokeh_chart(calib_plot, use_container_width=True)