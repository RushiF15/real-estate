import streamlit as st
from components.styling import apply_style
from src.data_loader import load_data
from src.sidebar import render_sidebar
from src.modeling import train_model,load_model

apply_style(); st.title("Future price scenario"); st.caption("Model estimates current listing prices from property attributes. Five-year figures are editable scenarios, not market forecasts.")
df=load_data()
if df is None: st.error("Dataset missing. Configure HOUSING_DATA_PATH."); st.stop()
d=render_sidebar(df)
bundle=load_model()
if bundle is None:
    st.warning("No trained current-price model is available yet. Train it here, or run `python train_model.py`.")
    if st.button("Train current-price model",type="primary"):
        with st.spinner("Training model and evaluating a held-out test set…"):
            try:
                model,metrics=train_model(df); st.success(f"Saved model. Holdout MAE ₹{metrics['mae']:.1f} lakh · R² {metrics['r2']:.3f}"); st.rerun()
            except Exception as e: st.error(f"Training failed: {e}")
    st.stop()
if d.empty: st.info("No properties match global filters."); st.stop()
st.success(f"Current-price model loaded · held-out MAE ₹{bundle['metrics']['mae']:.1f} lakh · R² {bundle['metrics']['r2']:.3f}")
idx=st.selectbox("Select a listing as the starting profile",d.index,format_func=lambda i:f"{d.loc[i,'City']} · {d.loc[i,'Property_Type']} · {d.loc[i,'BHK']} BHK · ₹{d.loc[i,'Price_in_Lakhs']:,.1f} lakh")
row=d.loc[[idx]]; current=float(bundle["model"].predict(row[bundle["features"]])[0]); st.metric("Estimated current price",f"₹{current:,.1f} lakh",help="Model estimate from property attributes, evaluated against a random held-out test split.")
rate=st.slider("Annual appreciation scenario (%)",min_value=-10.0,max_value=20.0,value=6.0,step=.5,help="User-entered assumption, compounded annually; it is not learned from this dataset.")
future=current*((1+rate/100)**5); st.metric("Five-year scenario value",f"₹{future:,.1f} lakh",f"{rate:.1f}% annual scenario")
st.caption(f"Calculation: model-estimated current price × (1 + {rate:.1f}% / 100)⁵. The data contains listing price and construction year, but no transaction dates or historical price series, so this is not a validated forecast of market appreciation.")
peers=d[(d.City==row.iloc[0].City)&(d.Property_Type==row.iloc[0].Property_Type)&(d.BHK==row.iloc[0].BHK)]
if not peers.empty: st.markdown("#### Similar active-filter listings"); st.caption(f"{len(peers):,} peer listings · median ₹{peers.Price_in_Lakhs.median():,.1f} lakh · median corrected rate ₹{peers.Price_per_SqFt_Clean.median():,.0f}/sq ft"); st.dataframe(peers[["City","Property_Type","BHK","Size_in_SqFt","Price_in_Lakhs","Price_per_SqFt_Clean"]].head(50),use_container_width=True,hide_index=True)
st.info("Current prices are listing prices, not verified sale prices. Model accuracy may vary across locations and property profiles. The five-year scenario is an assumption-based illustration.")
