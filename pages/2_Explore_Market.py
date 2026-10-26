import streamlit as st
import plotly.express as px
from components.styling import apply_style
from src.data_loader import load_data
from src.sidebar import render_sidebar
from components.charts import apply_chart_theme

apply_style(); st.title("Explore the market"); st.caption("Compare listing prices and property characteristics across the selected market.")
df=load_data()
if df is None: st.error("Dataset missing. Configure HOUSING_DATA_PATH or place the CSV in data/."); st.stop()
d=render_sidebar(df)
if d.empty: st.info("No properties match your filters."); st.stop()
c1,c2=st.columns(2)
with c1:
 g=d.groupby("State").agg(Median_price=("Price_in_Lakhs","median"),Listings=("ID","size")).reset_index(); st.plotly_chart(apply_chart_theme(px.scatter(g,x="Listings",y="Median_price",size="Listings",hover_name="State",title="State market comparison",labels={"Median_price":"Median price (₹ lakh)"},color="Median_price",color_continuous_scale="Blues")),use_container_width=True)
with c2:
 fig=px.box(d,x="Property_Type",y="Price_in_Lakhs",color="Property_Type",title="Price by property type",labels={"Price_in_Lakhs":"Price (₹ lakh)"}); st.plotly_chart(apply_chart_theme(fig),use_container_width=True)
fig=px.scatter(d.sample(min(12000,len(d)),random_state=42),x="Size_in_SqFt",y="Price_in_Lakhs",color="Property_Type",facet_col="Availability_Status",hover_data=["City","BHK","Price_per_SqFt_Clean"],title="Price versus property size",labels={"Size_in_SqFt":"Size (sq ft)","Price_in_Lakhs":"Price (₹ lakh)"}); st.plotly_chart(apply_chart_theme(fig),use_container_width=True)
c1,c2=st.columns(2)
with c1: st.plotly_chart(apply_chart_theme(px.box(d,x="BHK",y="Price_in_Lakhs",title="Price by BHK",labels={"Price_in_Lakhs":"Price (₹ lakh)"})),use_container_width=True)
with c2: st.plotly_chart(apply_chart_theme(px.box(d,x="Furnished_Status",y="Price_in_Lakhs",title="Furnishing versus price",labels={"Price_in_Lakhs":"Price (₹ lakh)"})),use_container_width=True)
c1,c2=st.columns(2)
with c1: st.plotly_chart(apply_chart_theme(px.scatter(d.sample(min(12000,len(d)),random_state=7),x="Age_of_Property",y="Price_in_Lakhs",color="Property_Type",title="Property age versus price",labels={"Age_of_Property":"Age (years; reference 2025)","Price_in_Lakhs":"Price (₹ lakh)"})),use_container_width=True)
with c2:
 g=d.groupby("State").Price_per_SqFt_Clean.median().nlargest(15).sort_values().reset_index(); st.plotly_chart(apply_chart_theme(px.bar(g,x="Price_per_SqFt_Clean",y="State",orientation="h",title="Median corrected price per sq ft by state",labels={"Price_per_SqFt_Clean":"₹ per sq ft"})),use_container_width=True)
c1,c2=st.columns(2)
with c1: st.plotly_chart(apply_chart_theme(px.box(d,x="Parking_Space",y="Price_in_Lakhs",color="Security",title="Price by parking and security",labels={"Price_in_Lakhs":"Price (₹ lakh)"})),use_container_width=True)
with c2: st.plotly_chart(apply_chart_theme(px.box(d,x="Public_Transport_Accessibility",y="Price_per_SqFt_Clean",title="Corrected rate by transport accessibility",labels={"Price_per_SqFt_Clean":"₹ per sq ft"})),use_container_width=True)
st.markdown("#### Amenity combinations"); amen=d.groupby("Amenities_Clean").agg(Median_rate=("Price_per_SqFt_Clean","median"),Listings=("ID","size")).query("Listings >= 30").sort_values("Median_rate",ascending=False).head(15).reset_index(); st.dataframe(amen,use_container_width=True,hide_index=True)
st.download_button("Download filtered properties",d.to_csv(index=False).encode(),"filtered_properties.csv","text/csv")
