"""
Hispanic Heritage Data Lab
World Bank historical explorer plus classroom function/calculus lab.

Run: python -m streamlit run app.py
"""
from __future__ import annotations

import io
import math
import re
import time
from datetime import date, datetime
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

APP_TITLE = "Hispanic Heritage Data Lab"
WORLD_BANK = "https://api.worldbank.org/v2"
COUNTRIES = {
    "Mexico": "MX", "Brazil": "BR", "Argentina": "AR", "Colombia": "CO",
    "Chile": "CL", "Peru": "PE", "Venezuela": "VE", "Ecuador": "EC", "Uruguay": "UY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn is expected to live if current age-specific mortality rates persist."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "Gross domestic product divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people, net", "Net number of migrants over the period. It is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, reported in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "Exports minus imports of goods and services as a percentage of GDP."),
}
# Constraints are intentionally modest: validate numeric data and obvious impossible values, not disputed historical series.
NONNEGATIVE = {"Unemployment, national estimate", "GDP per capita", "Crude birth rate",
               "Intentional homicide rate", "Total reserves including gold"}
CONTEXTS = {
    "Business": ("revenue", "currency units", "time", "years", "f(x)=a*x^2+b*x+c", "Revenue may grow or decline; the derivative is the change in revenue per time unit."),
    "Engineering": ("position", "meters", "time", "seconds", "f(x)=a*x^2+b*x+c", "For position, the first derivative is velocity and the second derivative is acceleration."),
    "Medical": ("concentration", "mg/L", "time", "hours", "f(x)=A*exp(-k*x)", "A concentration curve can model a changing measured level; this is illustrative, not medical advice."),
    "Sports science": ("position", "meters", "time", "seconds", "f(x)=a*sin(b*x)", "Position changes over time; derivatives describe velocity and acceleration."),
    "Biology": ("population", "organisms", "time", "days", "f(x)=K/(1+A*exp(-r*x))", "Population values and rates depend on assumptions; a simple curve is not a causal explanation."),
    "Bioengineering": ("concentration", "mg/L", "time", "hours", "f(x)=A*exp(-k*x)", "A modeled concentration and its rate can help explain changing system output."),
    "Economic forecasting": ("index", "index points", "time", "years", "f(x)=a*x^2+b*x+c", "Growth or decline describes the index; its derivative describes the rate of change."),
    "Weather": ("temperature", "°C", "time", "hours", "f(x)=A*sin(b*x+c)+d", "The curve is a simplified temperature pattern, not a weather forecast."),
    "Custom": ("output", "units", "input", "units", "Enter your own function", "Edit the context explanation to match your application."),
}
COURSES = ["Honors Algebra 2", "Precalculus", "Calculus"]

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
st.title("🌎 Hispanic Heritage Data Lab")
st.caption("Explore authentic World Bank historical data, fit models, and investigate functions from Algebra 2 through Calculus.")
st.info("Brazil is included as a Latin American comparison country. Its inclusion does not imply that every country shares the same history, institutions, or economic conditions.")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_indicator(country_code: str, indicator_code: str, start_year: int, end_year: int):
    url = f"{WORLD_BANK}/country/{country_code}/indicator/{indicator_code}"
    try:
        response = requests.get(url, params={"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["year", "value"]), f"World Bank returned no usable observation array. URL: {response.url}"
        rows = [{"year": int(item["date"]), "value": item.get("value")} for item in payload[1]
                if item.get("date") and str(item["date"]).isdigit()]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["year", "value"]), f"No observations returned. URL: {response.url}"
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
        return df, f"Retrieved {len(df)} non-null observations from the World Bank API. URL: {response.url}"
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "Request timed out. Check your connection and try again."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), f"World Bank request failed: {exc}"
    except (ValueError, TypeError, KeyError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not parse the World Bank response: {exc}"

def validate_table(df: pd.DataFrame, indicator_name: str):
    errors = []
    if not {"year", "value"}.issubset(df.columns):
        return ["Table must contain year and value columns."]
    years_raw = df["year"]
    years = pd.to_numeric(years_raw, errors="coerce")
    vals = pd.to_numeric(df["value"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        errors.append("Every year must be a whole-number integer.")
    if years.duplicated().any():
        errors.append("Duplicate years are not allowed.")
    if vals.isna().any() or not np.isfinite(vals.dropna()).all():
        errors.append("Every value must be numeric and finite; blanks are not valid in the editable table.")
    if indicator_name in NONNEGATIVE and (vals.dropna() < 0).any():
        errors.append(f"{indicator_name} cannot be negative in this validation.")
    if indicator_name == "Unemployment, national estimate" and (vals.dropna() > 100).any():
        errors.append("Unemployment as a percentage cannot exceed 100.")
    if indicator_name == "Life expectancy at birth" and ((vals.dropna() < 0) | (vals.dropna() > 125)).any():
        errors.append("Life expectancy must be between 0 and 125 years for this classroom validation.")
    if indicator_name == "Intentional homicide rate" and (vals.dropna() > 1000).any():
        errors.append("Homicide rate exceeds the app's plausibility check of 1,000 per 100,000.")
    return errors

def make_poly(df: pd.DataFrame, degree: int):
    clean = df[["year", "value"]].copy()
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna().sort_values("year")
    clean = clean.drop_duplicates("year")
    if len(clean) < degree + 1 or clean["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years; found {clean['year'].nunique()}.")
    base = int(clean["year"].min())
    t = clean["year"].to_numpy(dtype=float) - base
    y = clean["value"].to_numpy(dtype=float)
    coef = np.polynomial.polynomial.polyfit(t, y, degree)
    pred = np.polynomial.polynomial.polyval(t, coef)
    return clean, base, coef, pred

def poly_string(coef, base, xname="year"):
    terms = []
    for power, c in enumerate(coef):
        if abs(c) < 1e-12:
            continue
        term = f"{c:.8g}" if power == 0 else (f"{c:.8g}·t" if power == 1 else f"{c:.8g}·t^{power}")
        terms.append(term)
    return "f(t) = " + (" + ".join(terms).replace("+ -", "- ") if terms else "0") + f", where t = {xname} − {base}"

def parse_expression(text, x):
    """Safely parse a SymPy expression using a restricted symbol/function set."""
    cleaned = text.strip().replace("^", "**")
    if not cleaned:
        raise ValueError("Enter a function, such as x**2 + 3*x - 1.")
    if len(cleaned) > 180:
        raise ValueError("Function is too long; keep it under 180 characters.")
    allowed = {"x": x, "pi": sp.pi, "E": sp.E, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs}
    try:
        expr = sp.sympify(cleaned, locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not read that function. Use x, numbers, +, -, *, /, **, parentheses, and sin/cos/tan/exp/log/sqrt/abs.") from exc
    if expr.free_symbols - {x}:
        raise ValueError("Use x as the only variable.")
    if expr.has(sp.Integral, sp.Derivative, sp.Lambda):
        raise ValueError("Derivatives, integrals, and lambda expressions are not accepted as input.")
    return expr

def numeric_func(expr, x):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    def evaluate(arr):
        with np.errstate(all="ignore"):
            out = np.asarray(fn(arr), dtype=float)
        if out.ndim == 0:
            out = np.full_like(np.asarray(arr, dtype=float), float(out), dtype=float)
        return out
    return evaluate

def safe_eval(expr, xval):
    symbol = next(iter(expr.free_symbols), sp.Symbol("x"))
    fn = sp.lambdify(symbol, expr, modules=["numpy"])
    try:
        with np.errstate(all="ignore"):
            value = float(fn(float(xval)))
        return value if math.isfinite(value) else None
    except Exception:
        return None

def derivative_analysis(expr, x, lo, hi):
    d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
    f, f1, f2 = [numeric_func(e, x) for e in (expr, d1, d2)]
    grid = np.linspace(lo, hi, 1801)
    y1, y2 = f1(grid), f2(grid)
    critical = []
    roots1 = sp.calculus.util.function_range(d1, x, sp.Interval(lo, hi)) if False else None
    # Numeric candidate roots; verify finite values and classify using nearby derivative signs.
    for i in range(1, len(grid)-1):
        if np.isfinite(y1[i]) and abs(y1[i]) < 1e-5:
            xr = float(grid[i])
            if not critical or abs(xr-critical[-1]) > (hi-lo)/300:
                eps = max((hi-lo)/10000, 1e-5)
                left, right = f1(max(lo, xr-eps)), f1(min(hi, xr+eps))
                typ = "local minimum" if left < 0 < right else "local maximum" if left > 0 > right else "stationary point (not classified as an extremum)"
                critical.append((xr, safe_eval(expr, xr), typ))
    inflections = []
    for i in range(1, len(grid)-1):
        if np.isfinite(y2[i-1:i+2]).all() and y2[i-1] * y2[i+1] < 0:
            xr = float(grid[i])
            if not inflections or abs(xr-inflections[-1][0]) > (hi-lo)/300:
                inflections.append((xr, safe_eval(expr, xr)))
    valid = np.isfinite(f(grid))
    increasing, decreasing = [], []
    # Summarize sign intervals from the sampled first derivative.
    chunks = []
    start = None
    last = None
    for xx, yy in zip(grid, y1):
        sign = 1 if np.isfinite(yy) and yy > 1e-7 else -1 if np.isfinite(yy) and yy < -1e-7 else 0
        if sign != last:
            if start is not None and last in (-1, 1):
                chunks.append((start, float(xx), "increasing" if last == 1 else "decreasing"))
            start, last = float(xx), sign
    if start is not None and last in (-1, 1):
        chunks.append((start, hi, "increasing" if last == 1 else "decreasing"))
    finite_pairs = [(float(xx), float(yy)) for xx, yy in zip(grid, f(grid)) if np.isfinite(yy)]
    extrema = []
    if finite_pairs:
        extrema = [("absolute minimum", min(finite_pairs, key=lambda z:z[1])), ("absolute maximum", max(finite_pairs, key=lambda z:z[1]))]
    return d1, d2, critical, inflections, chunks, extrema

def create_plot(df, fitted, base, coef, sample_increment, prediction_years, prediction_values,
                single_year=None, single_value=None, title="Historical data and polynomial fit", unit=""):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    if not df.empty:
        ax.scatter(df["year"], df["value"], label="Observed data", s=28, zorder=3)
        ax.plot(df["year"], fitted, label="Polynomial fit at observed years", linestyle="--", linewidth=1.5)
        ax.axvline(df["year"].min(), linestyle=":", linewidth=1, label="Historical boundary")
        lo, hi = int(df["year"].min()), int(df["year"].max())
        curve_x = np.linspace(lo, max(hi, hi + 1), 700)
        curve_y = np.polynomial.polynomial.polyval(curve_x-base, coef)
        ax.plot(curve_x, curve_y, label="Smooth fitted curve", linewidth=2)
        sampled = df.iloc[::sample_increment]
        ax.scatter(sampled["year"], sampled["value"], marker="x", s=55, label=f"Sample markers (every {sample_increment} row(s))")
    if prediction_years and prediction_values:
        inrange_x, inrange_y, outside_x, outside_y = [], [], [], []
        lo = int(df["year"].min()) if not df.empty else -math.inf
        hi = int(df["year"].max()) if not df.empty else math.inf
        for xx, yy in zip(prediction_years, prediction_values):
            if yy is None or not np.isfinite(yy): continue
            (inrange_x if lo <= xx <= hi else outside_x).append(xx)
            (inrange_y if lo <= xx <= hi else outside_y).append(yy)
        if inrange_x: ax.scatter(inrange_x, inrange_y, marker="D", s=55, label="Calculated within fitted range")
        if outside_x: ax.scatter(outside_x, outside_y, marker="^", s=65, label="Extrapolated estimate")
    if single_year is not None and single_value is not None and np.isfinite(single_value):
        ax.scatter([single_year], [single_value], marker="*", s=160, label="Single-year estimate", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(unit or "Value")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def to_pdf(metadata, summary, chart_bytes, preview, results):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(APP_TITLE, styles["Title"]), Paragraph(metadata, styles["Normal"]), Spacer(1, 8)]
    for line in summary:
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 4))
    if chart_bytes:
        story.append(Spacer(1, 8))
        story.append(RLImage(io.BytesIO(chart_bytes), width=8.8*inch, height=4.2*inch))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Data preview", styles["Heading2"]))
    data = [["Year", "Value"]] + [[str(r.year), f"{r.value:.8g}"] for r in preview.itertuples(index=False)]
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]))
    story.append(tbl)
    if not results.empty:
        story.extend([Spacer(1,8), Paragraph("Calculated-year results", styles["Heading2"])])
        rdata = [list(results.columns)] + results.astype(str).values.tolist()
        rtbl = Table(rdata, repeatRows=1)
        rtbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]))
        story.append(rtbl)
    doc.build(story)
    return buf.getvalue()

with st.sidebar:
    st.header("Data controls")
    country_name = st.selectbox("Country", list(COUNTRIES.keys()), index=0)
    country_code = COUNTRIES[country_name]
    indicator_name = st.selectbox("Indicator", list(INDICATORS.keys()))
    indicator_code, unit, definition = INDICATORS[indicator_name]
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_increment = st.slider("Graph sample increment", 1, 10, 2)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    compare_country = st.selectbox("Comparison country", ["None"] + [c for c in COUNTRIES if c != country_name])
    printer_mode = st.checkbox("Printer-friendly view", value=False)
    if st.button("Reset to source", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("edit_") or key.startswith("editor_edit_") or key.startswith("calc_") or key.startswith("batch_") or key == "batch_result":
                del st.session_state[key]
        st.rerun()

current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
st.caption(f"Requested source window: {start_year}–{end_year} inclusive. Retrieval date: {date.today().isoformat()}.")
st.markdown(f"**Indicator definition:** {definition}  \n**Unit:** {unit}  \n**World Bank series:** [`{indicator_code}`]({WORLD_BANK}/country/{country_code}/indicator/{indicator_code}?format=json)")

with st.spinner("Retrieving official World Bank observations…"):
    source_df, retrieval_message = fetch_indicator(country_code, indicator_code, start_year, end_year)
st.write(retrieval_message)
if source_df.empty:
    st.error("No usable source data is available for this selection. The app will not invent replacement values. Change the selection or retry later.")
    st.stop()

edit_key = f"edit_{country_code}_{indicator_code}"
if edit_key not in st.session_state:
    st.session_state[edit_key] = source_df.copy()
edited = st.data_editor(st.session_state[edit_key], num_rows="dynamic", use_container_width=True, key=f"editor_{edit_key}",
                         column_config={"year": st.column_config.NumberColumn("Year", step=1, format="%d"), "value": st.column_config.NumberColumn(f"Value ({unit})")})
errors = validate_table(edited, indicator_name)
if errors:
    for error in errors: st.error(error)
    st.warning("Fix the editable table before fitting or exporting. The original API response remains separate and can be restored with Reset to source.")
    st.stop()
edited = edited.astype({"year": int, "value": float}).sort_values("year").reset_index(drop=True)
source_for_country = source_df.copy()
is_modified = not edited.equals(source_for_country.astype({"year": int, "value": float}).sort_values("year").reset_index(drop=True))
if is_modified:
    st.warning("USER-MODIFIED SCENARIO: results use the edited year/value table, not only the original source observations.")
st.metric("Valid observations", len(edited))
if not edited.empty:
    st.write(f"Actual coverage: **{edited.year.min()}–{edited.year.max()}** ({len(edited)} observations). Missing years and null source observations are not filled.")
st.caption(f"Source: [World Bank indicator API]({WORLD_BANK}/country/{country_code}/indicator/{indicator_code}?format=json) · [Indicator catalog](https://data.worldbank.org/indicator/{indicator_code})")

try:
    fit_df, base, coef, fitted_values = make_poly(edited, degree)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

r2 = r2_score(fit_df["value"], fitted_values) if len(fit_df) > 1 else float("nan")
mae = mean_absolute_error(fit_df["value"], fitted_values)
residuals = fit_df["value"].to_numpy() - fitted_values
mean_value = float(fit_df["value"].mean())
eqn = poly_string(coef, base)
left, right = st.columns([1.25, 1])
with left:
    st.subheader("Historical model")
    st.code(eqn)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("In-sample R²", f"{r2:.5f}")
    c2.metric("In-sample MAE", f"{mae:.5g} {unit}")
    c3.metric("Mean", f"{mean_value:.5g} {unit}")
    c4.metric("Base year", str(base))
    st.write(f"Degree: **{degree}** · Observations used: **{len(fit_df)}** · Fitted range: **{fit_df.year.min()}–{fit_df.year.max()}**")
    st.caption("These are in-sample fit metrics, not proof of causal explanation. Polynomial extrapolation can become implausible outside the observed range; residuals and MAE describe historical fit error, not a validated 95% prediction interval.")
with right:
    st.subheader("Residuals")
    resid_df = pd.DataFrame({"year": fit_df.year, "residual": residuals})
    st.dataframe(resid_df, use_container_width=True, height=180)

# Chronological holdout: reserve the last 20% (at least one) when enough points remain to fit degree d.
holdout_summary = None
min_train = degree + 1
holdout_n = max(1, int(round(len(fit_df)*.2)))
if len(fit_df) >= min_train + 2:
    train = fit_df.iloc[:-holdout_n]
    test = fit_df.iloc[-holdout_n:]
    if len(train) >= min_train:
        hc = np.polynomial.polynomial.polyfit(train.year.to_numpy()-base, train.value.to_numpy(), degree)
        hp = np.polynomial.polynomial.polyval(test.year.to_numpy()-base, hc)
        holdout_summary = f"Chronological holdout MAE: **{mean_absolute_error(test.value, hp):.5g} {unit}** on {len(test)} later observation(s); separate from in-sample metrics."
if holdout_summary: st.write(holdout_summary)
else: st.caption("Chronological holdout score unavailable: more observations are needed to retain at least degree + 1 training years.")

# Prediction controls and persistent results
st.subheader("Calculate values")
calc_col1, calc_col2 = st.columns(2)
with calc_col1:
    if "calc_year" not in st.session_state: st.session_state.calc_year = max(1950, min(2100, int(fit_df.year.max())))
    single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=int(st.session_state.calc_year), step=1)
    if st.button("Calculate Value"):
        st.session_state.calc_year = int(single_year)
        st.session_state.calc_result = (country_code, indicator_code, degree, hash(tuple(edited.year)), hash(tuple(np.round(edited.value, 8))), int(single_year))
    result_key = (country_code, indicator_code, degree, hash(tuple(edited.year)), hash(tuple(np.round(edited.value, 8))), int(single_year))
    if st.session_state.get("calc_result") and st.session_state.calc_result[:5] == result_key[:5]:
        val = float(np.polynomial.polynomial.polyval(single_year-base, coef))
        inside = fit_df.year.min() <= single_year <= fit_df.year.max()
        st.metric(f"Estimated value in {single_year}", f"{val:.7g} {unit}")
        st.write(f"Classification: **{'within' if inside else 'outside'} the actual fitted range ({fit_df.year.min()}–{fit_df.year.max()}).**")
        if single_year in set(fit_df.year):
            actual = float(fit_df.loc[fit_df.year == single_year, "value"].iloc[0])
            st.write(f"Observed source/edited value: **{actual:.7g} {unit}**. The model estimate at this year is not the same as the observation.")
        if not inside and (not np.isfinite(val) or abs(val) > max(1e6, abs(mean_value)*1e4)):
            st.warning("This extrapolation is numerically extreme or non-finite. It is shown without clipping; do not interpret it as a reliable forecast.")
    else:
        st.caption("Press Calculate Value to save and display a single-year estimate. Results reset when country, indicator, degree, or edited data changes.")
with calc_col2:
    batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
    batch_years = []
    invalid_batch = []
    if batch_mode == "Year range":
        range_cols = st.columns(3)
        with range_cols[0]: range_start = st.number_input("Start year", 1950, 2100, max(1950, min(2100, int(fit_df.year.min()))), key="batch_start")
        with range_cols[1]: range_end = st.number_input("End year", 1950, 2100, max(1950, min(2100, int(fit_df.year.max()))), key="batch_end")
        with range_cols[2]: range_step = st.slider("Range step", 1, 10, 1)
        if range_end < range_start: invalid_batch.append("End year must be greater than or equal to start year.")
        else: batch_years = list(range(int(range_start), int(range_end)+1, int(range_step)))
    else:
        custom_year_text = st.text_input("Comma-separated years", "2000, 2010, 2020, 2030")
        for token in custom_year_text.split(","):
            token = token.strip()
            if not token: continue
            try:
                number = float(token)
                if not number.is_integer() or not 1950 <= number <= 2100: invalid_batch.append(f"Invalid year: {token!r}. Years must be whole numbers from 1950 to 2100.")
                else: batch_years.append(int(number))
            except ValueError: invalid_batch.append(f"Invalid year: {token!r}.")
    batch_years = list(dict.fromkeys(batch_years))
    if invalid_batch:
        for msg in invalid_batch: st.error(msg)
    valid_batch = [y for y in batch_years if 1950 <= y <= 2100]
    batch_results = pd.DataFrame()
    if valid_batch and not invalid_batch:
        batch_values = np.polynomial.polynomial.polyval(np.array(valid_batch)-base, coef)
        batch_results = pd.DataFrame({"year": valid_batch, "estimate": batch_values, "unit": unit,
                                      "range_class": ["within fitted range" if fit_df.year.min() <= y <= fit_df.year.max() else "outside fitted range" for y in valid_batch]})
        st.dataframe(batch_results, use_container_width=True, height=190)
        st.session_state["batch_result"] = batch_results
    elif not invalid_batch:
        st.caption("Enter at least one valid year to calculate a batch.")

display_years = list(valid_batch)
display_values = list(np.polynomial.polynomial.polyval(np.array(display_years)-base, coef)) if display_years else []
fig = create_plot(fit_df, fitted_values, base, coef, sample_increment, display_years, display_values,
                  int(single_year) if st.session_state.get("calc_result") and st.session_state.calc_result[:5] == result_key[:5] else None,
                  float(np.polynomial.polynomial.polyval(single_year-base, coef)) if st.session_state.get("calc_result") and st.session_state.calc_result[:5] == result_key[:5] else None,
                  f"{country_name}: {indicator_name}", unit)
if horizon > 0:
    lo, hi = int(fit_df.year.min()), int(fit_df.year.max())
    extra_x = np.arange(hi+1, hi+horizon+1)
    extra_y = np.polynomial.polynomial.polyval(extra_x-base, coef)
    fig.axes[0].plot(extra_x, extra_y, linestyle=":", linewidth=2, label=f"Extrapolated display ({horizon} years)")
    fig.axes[0].legend(fontsize=8)
st.pyplot(fig, use_container_width=True)
chart_buf = io.BytesIO()
fig.savefig(chart_buf, format="png", dpi=150, bbox_inches="tight")
chart_bytes = chart_buf.getvalue()
plt.close(fig)

if compare_country != "None":
    st.subheader(f"Country comparison: {indicator_name}")
    compare_code = COUNTRIES[compare_country]
    comp_df, comp_msg = fetch_indicator(compare_code, indicator_code, start_year, end_year)
    st.caption(f"{compare_country}: {comp_msg}")
    if not comp_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(fit_df.year, fit_df.value, marker="o", label=f"{country_name} ({len(fit_df)} observations)")
        ax2.plot(comp_df.year, comp_df.value, marker=".", label=f"{compare_country} ({len(comp_df)} observations)")
        ax2.set_xlabel("Year"); ax2.set_ylabel(unit); ax2.set_title(f"{indicator_name}: country comparison")
        ax2.grid(True, alpha=.25); ax2.legend(); fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)
        st.caption("Lines may cover different years because source coverage and missing observations vary by country. Values use the same indicator definition and unit; do not infer causes from this comparison alone.")

st.divider()
st.header("Function and Calculus Lab")
course = st.radio("Course level", COURSES, horizontal=True)
context = st.selectbox("Context", list(CONTEXTS.keys()))
default_y, default_yunit, default_x, default_xunit, suggested, context_explanation = CONTEXTS[context]
context_desc = st.text_input("Context description", context_explanation)
xname = st.text_input("Independent variable name", default_x)
xunit = st.text_input("Independent variable unit", default_xunit)
yname = st.text_input("Dependent variable name", default_y)
yunit = st.text_input("Dependent variable unit", default_yunit)
function_source = st.radio("Function source", ["Use curve of best fit", "Suggested function", "Build from parameters", "Enter my own function"], horizontal=True)
x = sp.Symbol("x", real=True)
if function_source == "Use curve of best fit":
    # Convert the polynomial into expression with t = year - base so its fitted transformation is preserved.
    expr = sum(sp.Float(float(c), 12) * (x-base)**i for i,c in enumerate(coef))
    st.caption(f"Using the historical polynomial in {xname} with base year {base}. Its input variable is the calendar year.")
elif function_source == "Suggested function":
    st.caption(f"Suggested model: `{suggested}`. Replace the default expression if you want to customize it.")
    default_expr = "x**2 + 2*x + 1"
    if context == "Medical" or context == "Bioengineering": default_expr = "100*exp(-0.25*x)"
    elif context == "Biology": default_expr = "1000/(1 + 9*exp(-0.3*x))"
    elif context == "Weather" or context == "Sports science": default_expr = "10*sin(0.5*x) + 20"
    elif context == "Business" or context == "Economic forecasting": default_expr = "2*x**2 - 3*x + 10"
    elif context == "Engineering": default_expr = "0.5*x**2 + 2*x"
    else: default_expr = "x**2 + 2*x + 1"
    custom_expr = st.text_input("Suggested function expression in x", default_expr)
    try: expr = parse_expression(custom_expr, x)
    except ValueError as exc: st.error(str(exc)); st.stop()
elif function_source == "Build from parameters":
    shape = st.selectbox("Function family", ["Polynomial (quadratic)", "Rational", "Exponential", "Logarithmic", "Trigonometric"])
    a = st.number_input("Parameter a", value=1.0, step=0.5)
    b = st.number_input("Parameter b", value=1.0, step=0.5)
    c = st.number_input("Parameter c", value=0.0, step=0.5)
    if shape == "Polynomial (quadratic)": expr = sp.Float(a)*x**2 + sp.Float(b)*x + sp.Float(c)
    elif shape == "Rational": expr = (sp.Float(a)*x + sp.Float(c))/(sp.Float(b)*x + 1)
    elif shape == "Exponential": expr = sp.Float(a)*sp.exp(sp.Float(b)*x) + sp.Float(c)
    elif shape == "Logarithmic": expr = sp.Float(a)*sp.log(sp.Float(b)*x) + sp.Float(c)
    else: expr = sp.Float(a)*sp.sin(sp.Float(b)*x) + sp.Float(c)
    st.code(f"f(x) = {sp.sstr(expr)}")
else:
    custom_expr = st.text_input("Function expression", "x**3 - 3*x")
    try: expr = parse_expression(custom_expr, x)
    except ValueError as exc: st.error(str(exc)); st.stop()

st.write(f"**Function:** \\(f(x) = {sp.latex(expr)}\\)")
d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
if course == "Calculus":
    st.write(f"**First derivative:** \\(f'(x) = {sp.latex(d1)}\\)")
    st.write(f"**Second derivative:** \\(f''(x) = {sp.latex(d2)}\\)")
else:
    st.caption("Choose Calculus mode to reveal the full derivative analysis. This mode focuses on function behavior and graphical trends.")

domain_lo, domain_hi = st.slider("Analysis interval for x", -20.0, 20.0, (-5.0, 5.0), step=0.5)
if domain_hi <= domain_lo:
    st.error("The analysis interval must have a positive width."); st.stop()
point_x = st.slider("Current input value x", float(domain_lo), float(domain_hi), float((domain_lo+domain_hi)/2), step=max((domain_hi-domain_lo)/100, .01))
a_col,b_col = st.columns(2)
with a_col:
    interval_a = st.slider("Secant interval start", float(domain_lo), float(domain_hi), float(domain_lo), step=max((domain_hi-domain_lo)/100,.01))
with b_col:
    interval_b = st.slider("Secant interval end", float(domain_lo), float(domain_hi), float(domain_hi), step=max((domain_hi-domain_lo)/100,.01))
if interval_b < interval_a: interval_a, interval_b = interval_b, interval_a

# Domain validity mask; lambdified expressions yield NaN/inf at undefined points.
f, f1, f2 = [numeric_func(e, x) for e in (expr,d1,d2)]
gx = np.linspace(domain_lo, domain_hi, 1400)
gy = f(gx); gy1 = f1(gx); gy2 = f2(gx)
valid = np.isfinite(gy)
figf, axf = plt.subplots(figsize=(10,4.8))
# Plot contiguous valid sections separately to avoid connecting across discontinuities.
segments = []
start_idx = None
for i, ok in enumerate(valid):
    if ok and start_idx is None: start_idx = i
    if start_idx is not None and (not ok or i == len(valid)-1):
        end_idx = i if ok and i == len(valid)-1 else i-1
        if end_idx >= start_idx: segments.append((start_idx,end_idx))
        start_idx = None
for j,(s,e) in enumerate(segments):
    axf.plot(gx[s:e+1], gy[s:e+1], label="f(x)" if j==0 else None, linewidth=2)
show_first = st.checkbox("Show first derivative curve", value=(course=="Calculus"))
show_second = st.checkbox("Show second derivative curve", value=False)
if show_first:
    for j,(s,e) in enumerate(segments):
        axf.plot(gx[s:e+1], gy1[s:e+1], linestyle="--", label="f′(x)" if j==0 else None)
if show_second:
    for j,(s,e) in enumerate(segments):
        axf.plot(gx[s:e+1], gy2[s:e+1], linestyle=":", label="f″(x)" if j==0 else None)
fy = safe_eval(expr, point_x); slope = safe_eval(d1, point_x)
if fy is not None:
    axf.scatter([point_x],[fy],s=70,marker="o",label="Moving point")
    if slope is not None:
        tx = np.linspace(max(domain_lo,point_x-(domain_hi-domain_lo)*.18), min(domain_hi,point_x+(domain_hi-domain_lo)*.18),100)
        axf.plot(tx, fy+slope*(tx-point_x), linewidth=1.5, label="Tangent line")
fa, fb = safe_eval(expr, interval_a), safe_eval(expr, interval_b)
if fa is not None and fb is not None and interval_b != interval_a:
    secant_slope=(fb-fa)/(interval_b-interval_a)
    sx=np.linspace(interval_a,interval_b,100)
    axf.plot(sx,fa+secant_slope*(sx-interval_a), linestyle="-.", label="Secant line")
axf.axhline(0,linewidth=.7); axf.axvline(0,linewidth=.7)
axf.set_xlabel(f"{xname} ({xunit})"); axf.set_ylabel(f"{yname} ({yunit})")
axf.set_title(f"{context}: function and rate of change")
axf.grid(True,alpha=.25); axf.legend(fontsize=8); figf.tight_layout()
st.pyplot(figf, use_container_width=True)
st.caption("Graph uses gaps for non-finite values to avoid connecting across sampled discontinuities. Vertical asymptotes and domain restrictions can be subtle at finite resolution.")

if fy is None:
    st.error(f"f({point_x:.5g}) is undefined or non-finite in this model. Move the input point to a valid domain location.")
else:
    st.metric(f"{yname} at x = {point_x:.4g}", f"{fy:.6g} {yunit}")
    if slope is not None:
        st.write(f"Instantaneous rate: **{slope:.6g} {yunit}/{xunit}**. {context_desc}")
        if context == "Engineering" or "position" in yname.lower():
            accel = safe_eval(d2, point_x)
            st.write(f"Interpretation: the first derivative is velocity ({yunit}/{xunit}); the second derivative is acceleration ({yunit}/{xunit}²), currently {accel if accel is not None else 'undefined'}.")
        else:
            accel = safe_eval(d2, point_x)
            st.write(f"The second derivative at this point is {accel if accel is not None else 'undefined'} {yunit}/{xunit}² and describes how the rate itself is changing.")
if fa is not None and fb is not None:
    if interval_a == interval_b: st.write("Average rate of change is undefined because the interval endpoints are equal.")
    else: st.write(f"Average rate of change on [{interval_a:.4g}, {interval_b:.4g}]: **{(fb-fa)/(interval_b-interval_a):.6g} {yunit}/{xunit}**.")
else:
    st.write("Average rate of change is unavailable because one or both endpoints are outside the function's real-valued finite domain.")

if course == "Calculus":
    st.subheader("Calculus analysis (numerical over selected interval)")
    st.caption("Symbolic derivative formulas are exact for the parsed expression. Root locations and sign classifications below are numerical approximations on a grid and should be checked before making high-stakes claims.")
    d1, d2, critical, inflections, monotonic, extrema = derivative_analysis(expr,x,domain_lo,domain_hi)
    st.write("**Increasing/decreasing intervals (approximate):**")
    st.write("; ".join([f"{kind} on approximately [{a:.3g}, {b:.3g}]" for a,b,kind in monotonic]) or "No finite intervals classified.")
    st.write("**Critical-point candidates:**")
    if critical:
        for xx,yy,typ in critical: st.write(f"- x ≈ {xx:.6g}, f(x) ≈ {yy if yy is not None else 'undefined'}; {typ} (based on nearby derivative signs).")
    else: st.write("No critical-point candidates detected on this grid.")
    st.write("**Inflection candidates:**")
    verified=[]
    for xx,yy in inflections:
        eps=max((domain_hi-domain_lo)/10000,1e-5)
        l=safe_eval(d2,max(domain_lo,xx-eps)); r=safe_eval(d2,min(domain_hi,xx+eps))
        if l is not None and r is not None and l*r < 0: verified.append((xx,yy))
    if verified:
        for xx,yy in verified: st.write(f"- x ≈ {xx:.6g}, f(x) ≈ {yy if yy is not None else 'undefined'}; concavity changes across this point.")
    else: st.write("No verified concavity changes detected on the sampled interval.")
    if extrema:
        for kind,(xx,yy) in extrema: st.write(f"- Approximate {kind} on the selected closed interval: x ≈ {xx:.6g}, f(x) ≈ {yy:.6g}. Grid-based estimate; endpoints are included.")
    else: st.write("No finite values available for interval extrema.")
    st.write(f"**Domain restrictions:** denominator `{sp.denom(sp.together(expr))}` must not be zero; logarithm arguments must be positive; square-root radicands must be nonnegative over the reals. Check these restrictions for the chosen expression.")
    if expr.has(sp.tan): st.write("**Discontinuities:** tangent is undefined where its cosine denominator is zero.")
elif course == "Precalculus":
    st.subheader("Precalculus interpretation")
    st.write(f"The selected interval runs from {domain_lo:g} to {domain_hi:g} {xunit}. The graph's slope represents average change over an interval; the second derivative helps describe how the slope changes.")
else:
    st.subheader("Honors Algebra 2 interpretation")
    st.write(f"**Domain:** values of x in [{domain_lo:g}, {domain_hi:g}] where the expression is real and finite. **Range on this interval:** approximately {np.nanmin(gy) if np.isfinite(gy).any() else 'undefined'} to {np.nanmax(gy) if np.isfinite(gy).any() else 'undefined'} {yunit}.")
    st.write("**End behavior and trend:** inspect the left/right ends of the selected graph. Polynomial end behavior is governed by degree and leading coefficient; rational, logarithmic, exponential, and trigonometric models can behave differently.")
    if fa is not None and fb is not None and interval_a != interval_b:
        st.write(f"Average rate of change: {(fb-fa)/(interval_b-interval_a):.6g} {yunit}/{xunit}.")

# Playback controls: discrete deterministic state updates; no background timer required.
st.subheader("Simulation playback")
play_col1, play_col2, play_col3 = st.columns(3)
with play_col1: play = st.checkbox("Play", value=False)
with play_col2: speed = st.select_slider("Playback speed", options=["Slow","Normal","Fast"], value="Normal")
with play_col3:
    if st.button("Reset simulation"): st.session_state["sim_x"] = float(domain_lo); st.rerun()
if "sim_x" not in st.session_state or not domain_lo <= st.session_state.sim_x <= domain_hi:
    st.session_state.sim_x = float(point_x)
if play:
    step_size=(domain_hi-domain_lo)/100 * {"Slow":0.5,"Normal":1,"Fast":2}[speed]
    st.session_state.sim_x = domain_lo if st.session_state.sim_x + step_size > domain_hi else st.session_state.sim_x + step_size
    st.caption(f"Playback advanced to x ≈ {st.session_state.sim_x:.4g}. Toggle Play off to pause. Streamlit reruns on interaction; browser refresh may reset this lightweight simulation.")
    st.button("Advance frame", on_click=lambda: None, help="Interact with Play to advance one frame per rerun.")
else:
    st.caption("Paused. The main graph's moving point follows the Current input value slider; playback state is shown here. Use Reset simulation to return to the interval start.")

# Contextual simulation panel; visual is schematic and scaled from the function values.
sim_value = safe_eval(expr, st.session_state.get("sim_x", point_x))
st.write(f"**Simulation input:** x = {st.session_state.get('sim_x', point_x):.4g} {xunit}; modeled output = {sim_value if sim_value is not None else 'undefined'} {yunit}.")
if context in ("Engineering","Sports science"):
    st.markdown("**Visual simulation: moving object**")
    st.caption("Schematic marker position is scaled to the finite function values in the selected interval; it is not a real-world trajectory unless units and model assumptions support that use.")
    finite_vals=gy[np.isfinite(gy)]
    if sim_value is not None and len(finite_vals):
        ymin,ymax=float(finite_vals.min()),float(finite_vals.max()); frac=.5 if ymax==ymin else min(1,max(0,(sim_value-ymin)/(ymax-ymin)))
        st.progress(frac, text=f"Position indicator ({frac*100:.0f}% of displayed vertical scale)")
elif context in ("Medical","Bioengineering"):
    st.markdown("**Visual simulation: concentration indicator**")
    st.caption("Illustrative model only. Do not use this graph for dosing, diagnosis, or treatment decisions.")
    st.metric("Modeled concentration", f"{sim_value:.5g} {yunit}" if sim_value is not None else "Undefined")
elif context == "Biology":
    st.markdown("**Visual simulation: population / cell colony**")
    st.caption("Bar length is normalized to the displayed interval, not a count of actual organisms.")
    finite_vals=gy[np.isfinite(gy)]
    if sim_value is not None and len(finite_vals):
        frac=.5 if finite_vals.max()==finite_vals.min() else min(1,max(0,(sim_value-finite_vals.min())/(finite_vals.max()-finite_vals.min())))
        st.progress(frac,text=f"Relative modeled population ({frac*100:.0f}%)")
elif context == "Weather":
    st.markdown("**Visual simulation: thermometer**")
    st.caption("The fill is normalized to the displayed graph range; the selected function is not a forecast.")
    finite_vals=gy[np.isfinite(gy)]
    if sim_value is not None and len(finite_vals):
        frac=.5 if finite_vals.max()==finite_vals.min() else min(1,max(0,(sim_value-finite_vals.min())/(finite_vals.max()-finite_vals.min())))
        st.progress(frac,text=f"Temperature indicator ({frac*100:.0f}% of scale)")
elif context == "Business" or context == "Economic forecasting":
    st.markdown("**Visual simulation: business/economic indicator**")
    if sim_value is not None: st.metric(f"Modeled {yname}", f"{sim_value:.5g} {yunit}")
elif context == "Custom":
    st.markdown("**Visual simulation: custom output indicator**")
    if sim_value is not None: st.metric(yname, f"{sim_value:.5g} {yunit}")
st.caption("Prosthetic-leg walking is a possible classroom application of periodic position/angle data, but this app does not simulate medical-device mechanics or recommend a prosthesis.")

# Exports reflect current selection, editable data, model, and batch results.
st.divider()
st.header("Export current analysis")
metadata = f"Country: {country_name} ({country_code}); Indicator: {indicator_name} ({indicator_code}); retrieval date: {date.today().isoformat()}; scenario: {'user-modified' if is_modified else 'source-based'}; coverage: {fit_df.year.min()}–{fit_df.year.max()}."
summary_lines = [
    f"<b>Source and definition:</b> {definition} Unit: {unit}. World Bank API: {WORLD_BANK}/country/{country_code}/indicator/{indicator_code}",
    f"<b>Model:</b> {eqn}. Degree {degree}, base year {base}.",
    f"<b>In-sample metrics:</b> R² = {r2:.6g}; MAE = {mae:.6g} {unit}; mean = {mean_value:.6g} {unit}; n = {len(fit_df)}.",
    f"<b>Residual interpretation:</b> residuals are observed minus fitted values. Fit metrics summarize in-sample fit and do not establish causation.",
    "<b>Limitations:</b> Historical observations may be missing; polynomial extrapolation can become implausible. Estimates are not validated prediction intervals. Do not infer historical causes from curve shape alone.",
]
csv_bytes = fit_df.to_csv(index=False).encode("utf-8")
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    pd.DataFrame([{"country":country_name,"country_code":country_code,"indicator":indicator_name,"indicator_code":indicator_code,"unit":unit,"retrieval_date":date.today().isoformat(),"scenario":"user-modified" if is_modified else "source-based","degree":degree,"base_year":base,"equation":eqn,"in_sample_r2":r2,"in_sample_mae":mae,"mean":mean_value,"observations":len(fit_df),"coverage":f"{fit_df.year.min()}-{fit_df.year.max()}"}]).to_excel(writer,index=False,sheet_name="Summary")
    fit_df.assign(fitted_value=fitted_values,residual=residuals).to_excel(writer,index=False,sheet_name="Working Data")
    (batch_results if not batch_results.empty else pd.DataFrame(columns=["year","estimate","unit","range_class"])).to_excel(writer,index=False,sheet_name="Calculated Years")
pdf_bytes = to_pdf(metadata, summary_lines, chart_bytes, fit_df.head(20), batch_results)
e1,e2,e3 = st.columns(3)
with e1: st.download_button("Download CSV", csv_bytes, file_name="historical_working_data.csv", mime="text/csv", use_container_width=True)
with e2: st.download_button("Download Excel workbook", xlsx_buf.getvalue(), file_name="hispanic_heritage_data_lab.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
with e3: st.download_button("Download PDF report", pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf", use_container_width=True)

if printer_mode:
    st.divider()
    st.subheader("Printer-friendly report panel")
    st.write(metadata)
    st.write(f"Equation: {eqn}")
    st.write(f"In-sample R²: {r2:.5f} | In-sample MAE: {mae:.5g} {unit} | n = {len(fit_df)}")
    st.write("Limitations: Polynomial extrapolation is uncertain and may be implausible. MAE/residuals are not a 95% prediction interval.")
    st.dataframe(fit_df.head(15), use_container_width=True)
