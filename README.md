# Real Estate Investment Advisor

A Streamlit dashboard for exploring the supplied Indian housing listings, comparing property markets, reviewing question-led EDA, and evaluating property profiles. The dashboard keeps descriptive comparisons separate from validated investment outcomes and future market forecasts.

## Data

The app expects `india_housing_prices_cleaned.csv` or `india_housing_prices_cleaned (1).csv`. It searches `data/` and the project root. To use another path, set `HOUSING_DATA_PATH` before launching:

```powershell
$env:HOUSING_DATA_PATH = 'D:\path\to\india_housing_prices_cleaned.csv'
```

The source data is never modified. The supplied file has 250,000 rows and all 25 expected columns, with no missing cells or duplicate rows. `Age_of_Property` is consistent with `2025 - Year_Built`. The source `Price_per_SqFt` values are 0–0.99 and are not treated as a valid rate. `Price_per_SqFt_Clean` is used for descriptive comparisons and is excluded from model inputs because it is derived from price. 116,304 rows have `Floor_No > Total_Floors`; they are retained and flagged in the overview. `Locality` values are identifiers such as `Locality_84`, not human-readable locations. No transaction dates, rent, yield, crime, infrastructure score, or verified investment outcomes are supplied.

## Install and run

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux activate with `source .venv/bin/activate`. Main command: `streamlit run app.py`.

## Pages

- **Overview:** filtered KPI cards, distributions, market comparisons, computed insights and data-quality notes.
- **Explore Market:** city/state, type, size, BHK, age, furnishing, parking, security, transport and amenity comparisons, with filtered CSV download.
- **Question-wise EDA:** 20 questions in five expandable categories, each with guided subquestions, filtered computed answers, appropriate Plotly charts, supporting statistics and CSV downloads. Group comparisons are descriptive; schools/hospitals and transport associations are not causal claims. Construction year is not a price time series.
- **Investment Advisor:** similar listing comparisons plus an exploratory fit score. The score averages 11 equally weighted peer-relative checks (price efficiency against peer median rate/price, school count, hospital count, transport, parking, security, furnishing, facing, owner type, availability, and selected amenities). It is not a classifier, validated probability, or investment recommendation.
- **Future Price:** current listing price model estimate and editable five-year compound-growth scenario. The selected annual appreciation assumption is visible and user-controlled. It is not a validated forecast.
- **Model Insights:** saved training metrics, actual vs predicted chart, feature importances, leakage check and caveats.

## Training

Train the reproducible current listing-price pipeline:

```powershell
python train_model.py
```

It saves `models/current_price_pipeline.joblib`, uses an 80/20 random split with seed 42, and reports MAE, RMSE and R² on held-out data. Model inputs exclude `Price_in_Lakhs`, both price-per-square-foot columns, ID and raw amenities. `Amenities_Clean` and property attributes are used. The model predicts current supplied listing price only. Random split scores do not establish performance on future dates or unseen locations.

## Structure

```text
app.py                  Overview and application entry
pages/                  Explore, EDA, investment, scenario and model pages
components/             Shared styling, metrics and Plotly charts
src/                    Data validation, filters, analysis and ML pipeline
tests/                  Automated unit and edge-case tests
train_model.py          Reproducible training entry point
```

Run checks with `python -m pytest`. Tests use small synthetic fixtures only to test code paths; the dashboard itself does not substitute fabricated rows for missing real data.
