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
st.title("FTIR Spectra Baseline Correction, Normalization & Quantification")

st.sidebar.title("FTIR Spectra Settings")

uploaded_files = st.sidebar.file_uploader("Calibrant spectra (columns: wavenumber, absorbance)", type="csv", accept_multiple_files=True)
uploaded_analyt = st.sidebar.file_uploader("Analyte spectra (columns: wavenumber, absorbance)", type="csv", accept_multiple_files=True)

if uploaded_files and not isinstance(uploaded_files, list):
    uploaded_files = [uploaded_files]
if uploaded_analyt and not isinstance(uploaded_analyt, list):
    uploaded_analyt = [uploaded_analyt]

# Smoothness (Lambda) Selection
use_large_range = st.sidebar.checkbox("Use larger smoothness range (1,000,000 - 10,000,000,000)")
if use_large_range:
    lam = st.sidebar.slider("Smoothness (λ)", min_value=100000, max_value=10000000000, value=10000000, step=100000)
else:
    lam = st.sidebar.slider("Smoothness (λ)", min_value=1000, max_value=100000, value=10000, step=1000)

lam_value = lam * 1e3
st.sidebar.write(f"Lambda (Smoothness): {lam_value:.0e}")

p = st.sidebar.slider("Asymmetry (p)", min_value=0.0001, max_value=0.01, value=0.001, step=0.00005)
st.sidebar.write(f"p (Asymmetry): {p:.4f}")

# Peak area range
st.sidebar.subheader("Peak Area Integration Range")
wn_min = st.sidebar.number_input("Wavenumber Min", value=1000)
wn_max = st.sidebar.number_input("Wavenumber Max", value=1800)

# Normalization Settings
st.sidebar.subheader("Normalization Settings")
normalize = st.sidebar.checkbox("Activate normalization?")
norm_min = st.sidebar.number_input("Normalization Wavenumber Min", value=800)
norm_max = st.sidebar.number_input("Normalization Wavenumber Max", value=3500)

# --- Main Processing ---
area_results = []
analyte_areas = []
plots = []
palette = Category10[10]

# --- Process Calibrants ---
if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        df = pd.read_csv(file, skiprows=1)
        df = df.sort_values(by=df.columns[0], ascending=False)

        wn = df.iloc[:, 0].values
        absorbance = df.iloc[:, 1].values

        # Baseline correction
        baseline = baseline_als(absorbance, lam, p)
        corrected = absorbance - baseline

        # Normalization
        if normalize:
            idx_min = np.argmin(np.abs(wn - norm_min))
            idx_max = np.argmin(np.abs(wn - norm_max))
            if idx_min > idx_max:
                idx_min, idx_max = idx_max, idx_min
            ymax = np.max(corrected[idx_min:idx_max])
            if ymax != 0:
                corrected = corrected / ymax

        # Area under peak
        mask = (wn >= wn_min) & (wn <= wn_max)
        wn_sel = wn[mask]
        corr_sel = corrected[mask]
        area = abs(np.trapz(np.clip(corr_sel, a_min=0, a_max=None), wn_sel))
        area_results.append((file.name, area))

        # Plot
        fig = figure(title=f"Calibrant: {file.name}", width=700, height=300, x_axis_label="Wavenumber (cm⁻¹)", y_axis_label="Absorbance")
        fig.x_range.flipped = True  # Flip X-axis
        fig.y_range.flipped = True  # Flip Y-axis (abs=0 at top)

        fig.line(wn, absorbance, line_width=2, color="#CCCCCC", legend_label="Original")
        fig.line(wn, baseline, line_dash='dotted', color='gray', legend_label="Baseline")
        fig.line(wn, corrected, line_width=2, color=palette[idx % 10], legend_label="Corrected")

        source = ColumnDataSource(data={'x': wn_sel, 'lower': np.zeros_like(wn_sel), 'upper': corr_sel})
        band = Band(base='x', lower='lower', upper='upper', source=source, fill_alpha=0.3, level='underlay', line_width=0, fill_color=palette[idx % 10])
        fig.add_layout(band)

        fig.legend.location = "bottom_left"

        plots.append(fig)

# --- Process Analytes ---
if uploaded_analyt:
    for idx, file in enumerate(uploaded_analyt):
        df = pd.read_csv(file, skiprows=1)
        df = df.sort_values(by=df.columns[0], ascending=False)

        wn = df.iloc[:, 0].values
        absorbance = df.iloc[:, 1].values

        baseline = baseline_als(absorbance, lam, p)
        corrected = absorbance - baseline

        if normalize:
            idx_min = np.argmin(np.abs(wn - norm_min))
            idx_max = np.argmin(np.abs(wn - norm_max))
            if idx_min > idx_max:
                idx_min, idx_max = idx_max, idx_min
            ymax = np.max(corrected[idx_min:idx_max])
            if ymax != 0:
                corrected = corrected / ymax

        mask = (wn >= wn_min) & (wn <= wn_max)
        wn_sel = wn[mask]
        corr_sel = corrected[mask]
        area = abs(np.trapz(np.clip(corr_sel, a_min=0, a_max=None), wn_sel))
        analyte_areas.append((file.name, area))

        fig = figure(title=f"Analyte: {file.name}", width=700, height=300, x_axis_label="Wavenumber (cm⁻¹)", y_axis_label="Absorbance")
        fig.x_range.flipped = True
        fig.y_range.flipped = True

        fig.line(wn, absorbance, line_width=2, color="#CCCCCC", legend_label="Original")
        fig.line(wn, baseline, line_dash='dotted', color='gray', legend_label="Baseline")
        fig.line(wn, corrected, line_width=2, color='red', legend_label="Corrected")

        source = ColumnDataSource(data={'x': wn_sel, 'lower': np.zeros_like(wn_sel), 'upper': corr_sel})
        band = Band(base='x', lower='lower', upper='upper', source=source, fill_alpha=0.3, level='underlay', fill_color='red')
        fig.add_layout(band)

        fig.legend.location = "bottom_left"

        plots.append(fig)

# --- Display Spectra ---
if plots:
    st.bokeh_chart(column(*plots), use_container_width=True)

# --- Display Integrated Areas ---
if area_results:
    st.subheader("Calibrant Areas (Integrated)")
    for name, area in area_results:
        st.write(f"**{name}**: {area:.4f} absorbance·cm⁻¹")

if analyte_areas:
    st.subheader("Analyte Areas (Integrated)")
    for name, area in analyte_areas:
        st.write(f"**{name}**: {area:.4f} absorbance·cm⁻¹")

# --- Quantification Section ---
if uploaded_files and len(area_results) >= 3:
    st.subheader("Quantification Input (for Calibrants)")

    concentrations = []
    with st.form("quantification_form"):
        for name, _ in area_results:
            concentration = st.number_input(f"Concentration for **{name}** (same units)", min_value=0.0, step=0.01, key=name)
            concentrations.append(concentration)
        submitted = st.form_submit_button("Run Quantification")

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
        analyte_concentrations = []
        for name, area in analyte_areas:
            conc = (area - intercept) / slope
            analyte_concentrations.append(conc)
            st.write(f"**{name}** → {conc:.4f} (same units)")

        conc_array = np.array(analyte_concentrations)
        mean_conc = conc_array.mean()
        std_conc = conc_array.std()

        st.markdown(f"### Final Result: {mean_conc:.4f} ± {std_conc:.4f} (mean ± std)")

        quant_plot = figure(title="Quantification Curve", x_axis_label="Concentration", y_axis_label="Integrated Area", width=700, height=400)
        quant_plot.circle(x.flatten(), y.flatten(), size=10, color="blue", legend_label="Calibrants")
        x_fit = np.linspace(min(x)[0], max(x)[0], 100).reshape(-1, 1)
        y_fit = model.predict(x_fit)
        quant_plot.line(x_fit.flatten(), y_fit.flatten(), color="black", line_width=2, legend_label=f"Fit: y = {intercept:.2f} + {slope:.2f}·x | R² = {r2:.4f}")

        for name, area in analyte_areas:
            conc = (area - intercept) / slope
            quant_plot.square([conc], [area], size=12, color="red", legend_label="Analyte")

        quant_plot.legend.location = "top_left"
        quant_plot.legend.label_text_font_size = "10pt"

        st.bokeh_chart(quant_plot, use_container_width=True)

# --- If nothing uploaded yet ---
if not uploaded_files and not uploaded_analyt:
    st.warning("""
    Please upload at least one spectrum file to continue.  
    ➔ The file should have **one line of header** only (column names).  
    ➔ **Remove additional headers and footers** before uploading.  
    """)
