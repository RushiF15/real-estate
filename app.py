import streamlit as st

from components.styling import apply_style
from src.data_loader import load_data, validate_data
from src.sidebar import render_sidebar
from components.metrics import metric_card
from components.charts import price_histogram, count_bar, categorical_bar
from src.analysis import overview_insights

st.set_page_config(page_title="Property Intelligence | Real Estate Advisor", page_icon="▦", layout="wide", initial_sidebar_state="expanded")
apply_style()
df = load_data()
if df is None:
    st.error("Housing data was not found. Place `india_housing_prices_cleaned.csv` in `data/`, the project root, or set `HOUSING_DATA_PATH`.")
    st.stop()
report = validate_data(df)
filtered = render_sidebar(df)

st.markdown('<div class="eyebrow">REAL ESTATE INVESTMENT ADVISOR</div>', unsafe_allow_html=True)
st.title("Property intelligence, made practical")
st.markdown("Explore Indian housing data, compare markets and assess property profiles with transparent, data-led analysis.")
st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** properties · Descriptive figures below use the current filters.")
if report["bad_floors"]:
    st.warning(f"Data quality: {report['bad_floors']:,} records ({report['bad_floors']/len(df):.1%}) have Floor_No greater than Total_Floors. Records are retained; interpret floor-related comparisons with care.")
if filtered.empty:
    st.info("No properties match the selected filters. Adjust or reset filters in the sidebar.")
    st.stop()

c1,c2,c3,c4,c5,c6=st.columns(6)
metric_card(c1,"Properties",f"{len(filtered):,}","Matching selected filters")
metric_card(c2,"Average price",f"₹{filtered.Price_in_Lakhs.mean():,.1f}L","Lakhs")
metric_card(c3,"Median price",f"₹{filtered.Price_in_Lakhs.median():,.1f}L","Robust midpoint")
metric_card(c4,"Average size",f"{filtered.Size_in_SqFt.mean():,.0f}","Square feet")
metric_card(c5,"Avg. corrected rate",f"₹{filtered.Price_per_SqFt_Clean.mean():,.0f}","Per sq ft")
metric_card(c6,"Cities",f"{filtered.City.nunique():,}","Represented in selection")

st.markdown("<div class='section-heading'>Market snapshot</div>",unsafe_allow_html=True)
a,b=st.columns([1.3,1])
with a: st.plotly_chart(price_histogram(filtered),use_container_width=True)
with b: st.plotly_chart(count_bar(filtered,"Property_Type","Properties by type"),use_container_width=True)
c,d=st.columns(2)
with c: st.plotly_chart(categorical_bar(filtered,"State","Price_in_Lakhs","Average price by state",aggregation="mean",top=12),use_container_width=True)
with d: st.plotly_chart(count_bar(filtered,"Availability_Status","Availability mix"),use_container_width=True)
c,d=st.columns(2)
with c: st.plotly_chart(count_bar(filtered,"Furnished_Status","Furnishing status"),use_container_width=True)
with d: st.plotly_chart(count_bar(filtered,"City","Top cities by listing count",top=12),use_container_width=True)

st.markdown("<div class='section-heading'>What stands out</div>",unsafe_allow_html=True)
for insight in overview_insights(filtered): st.markdown(f"<div class='insight-card'>{insight}</div>",unsafe_allow_html=True)
st.markdown("<div class='section-heading'>Data notes</div>",unsafe_allow_html=True)
st.caption(f"Loaded {len(df):,} rows · {report['duplicate_rows']:,} duplicate rows · {report['missing_cells']:,} missing cells. `Age_of_Property` matches 2025 − `Year_Built` in {report['age_mismatch']:,} rows. `Price_per_SqFt` is not used because its values appear invalid; corrected price/sq ft is calculated from price and area.")
