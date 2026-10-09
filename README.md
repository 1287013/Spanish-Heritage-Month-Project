# Hispanic Heritage Data Lab

A Streamlit classroom app combining official World Bank historical indicators, polynomial curve fitting, editable scenarios, exports, and an interactive function/calculus lab.

## Requirements
- Python 3.11 recommended
- Internet access for the World Bank API
- Dependencies pinned in `requirements.txt`

The dependency versions are pinned as a compatibility target; this workspace did not execute a full browser-based Streamlit acceptance test. Do not describe the app as production-certified. Run the checks below in your environment.

## Run locally

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and add `app.py`, `requirements.txt`, `README.md`, and `sources.md` at its root.
2. Commit and push the files to GitHub.
3. In Streamlit Community Cloud, create an app from that repository, select the branch, and set the main file path to `app.py`.
4. Deploy. No API key is required for the public World Bank API.
5. Review the deployment logs if dependencies fail to install or the app raises an exception.

## Deploy on Render
1. Push the four files to a GitHub repository.
2. Create a **Python Web Service** connected to the repository.
3. Build command:

```bash
pip install -r requirements.txt
```

4. Start command (one line):

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

5. Deploy and inspect Render logs. If `$PORT` is not expanded by a particular shell configuration, use Render's standard environment-variable syntax for the selected runtime.

## What the app does
- Requests ten World Bank series for nine Latin American countries, with a rolling 70-year request window ending last year.
- Shows observed coverage and the actual count of non-null observations; missing values are not fabricated.
- Separates cached source data from an editable year/value table and labels changes as a user-modified scenario.
- Fits a polynomial in centered time, `t = year - base year`, and reports the equation, in-sample R², in-sample MAE, residuals, count, historical range, mean, and a chronological holdout MAE when enough observations remain.
- Offers single-year and batch estimates, a country comparison, plot sample controls, horizon control, and CSV/XLSX/PDF exports.
- Includes course modes and a function lab supporting polynomial, rational, exponential, logarithmic, and trigonometric expressions; Calculus mode displays symbolic derivatives and numerical analyses.

## Source accuracy and limitations
- Source: World Bank API, linked in the UI and in `sources.md`.
- GDP per capita is not average wealth or household income.
- Crude birth rate is a rate, not a count.
- Net migration is not an emigration count.
- Missing observations are left missing. A failed API request is shown as an error, never replaced with random data.
- Polynomial fits are descriptive approximations. High-degree fits may oscillate, and extrapolation is especially unreliable. R² and MAE are in-sample metrics; they are not validated prediction intervals and do not imply causation.
- Function calculus is exact only for the symbolic derivative expressions. Root locations, extrema, and interval classifications are numerical approximations sampled over the selected interval. Check results independently for coursework.
- Context examples are simplified mathematical models, not medical, engineering, weather, or financial advice. The app does not infer historical causes.

## Verification checklist
1. Confirm country/indicator selections change source URL, definition, unit, chart, model, and export metadata.
2. Compare the UI values with the linked World Bank API; verify coverage and observation count.
3. Edit a value and a year; confirm the scenario warning and all outputs/exports update. Click **Reset to source** to restore the fetched values.
4. Try a duplicate year, fractional year, nonnumeric value, negative value for a nonnegative series, and unemployment above 100; confirm validation blocks fitting.
5. Select a degree with too few distinct years; confirm the app explains the minimum `degree + 1` requirement.
6. Press **Calculate Value** and compare that estimate with the same year in the batch table.
7. Try custom inputs such as `x**3 - 3*x`, `1/x`, `log(x)`, `exp(-x)`, and `sin(x)`; verify undefined values and discontinuities are not connected as a single curve.
8. Move the point and interval sliders; confirm displayed values and secant/tangent lines update.
9. Toggle derivative curves, course modes, contexts, sample increment, forecast horizon, comparison country, and printer-friendly mode.
10. Download CSV, Excel, and PDF; confirm XLSX opens with Summary, Working Data, and Calculated Years sheets, and PDF contains the chart and report sections.
11. For API failures, check internet access, World Bank response status, and Streamlit logs. For deployment errors, inspect dependency installation output and Python tracebacks.

## Team explanation: functions and data flow
The sidebar selects country, indicator, polynomial degree, graph sampling, forecast display horizon, and comparison country. A cached HTTP request retrieves the World Bank series. The editable table is validated separately; fitting proceeds only when its year/value pairs are valid. The model uses centered time `t = year - base year`, which improves numerical conditioning compared with using calendar years directly. Predictions reuse this same centered transformation. In-sample R² measures explained variation on the fitting data; MAE is the mean absolute residual size. The chronological holdout fits earlier years and evaluates later ones when sufficient observations exist. Batch and single-year outputs use the same coefficients.

The function lab parses a restricted mathematical expression with SymPy, derives first and second derivatives symbolically, and uses NumPy evaluation for plotting. Numerical root and extrema detection uses a finite grid, so it may miss closely spaced roots, unusual singularities, or behavior between sample points. Domain restrictions should be checked algebraically. The simulation indicators normalize output values to the displayed interval and are schematic, not physical validation.
