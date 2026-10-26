import streamlit as st
import pandas as pd
import plotly.express as px
from components.styling import apply_style
from components.charts import apply_chart_theme
from src.data_loader import load_data
from src.modeling import load_model,train_model,feature_importance,leakage_check

apply_style(); st.title("Model insights"); st.caption("Transparent evaluation of the current listing-price estimation pipeline.")
df=load_data()
if df is None: st.error("Dataset missing. Configure HOUSING_DATA_PATH."); st.stop()
bundle=load_model()
if bundle is None:
    st.info("No trained pipeline found in `models/current_price_pipeline.joblib`. Train reproducibly with `python train_model.py`.")
    if st.button("Train model now",type="primary"):
        with st.spinner("Training and evaluating…"):
            try: train_model(df); st.rerun()
            except Exception as e: st.error(f"Training failed: {e}")
    st.stop()
m=bundle["metrics"]
st.markdown("**Model:** Random forest regression · **Target:** current listing price (`Price_in_Lakhs`) · **Split:** 80% train / 20% test, random_state 42")
a,b,c,d=st.columns(4); a.metric("Rows used",f"{m['rows']:,}"); b.metric("Test MAE",f"₹{m['mae']:.1f}L"); c.metric("Test RMSE",f"₹{m['rmse']:.1f}L"); d.metric("Test R²",f"{m['r2']:.3f}")
fig=px.scatter(x=m["y_test"],y=m["predictions"],labels={"x":"Actual price (₹ lakh)","y":"Predicted price (₹ lakh)"},title="Held-out actual versus predicted prices",opacity=.45); lo=min(min(m["y_test"]),min(m["predictions"])); hi=max(max(m["y_test"]),max(m["predictions"])); fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi,line={"dash":"dash","color":"#B91C1C"}); st.plotly_chart(apply_chart_theme(fig),use_container_width=True)
imp=feature_importance(bundle).head(20).sort_values("Importance"); st.plotly_chart(apply_chart_theme(px.bar(imp,x="Importance",y="Feature",orientation="h",title="Top model feature importances")),use_container_width=True)
st.markdown("#### Leakage checks")
leaks=leakage_check(bundle["features"]); st.success("No known target-derived price fields are included as predictors." if not leaks else "Potential leakage columns detected: "+", ".join(leaks))
st.caption("Excluded: target price, original suspicious Price_per_SqFt, corrected Price_per_SqFt_Clean (derived directly from target), ID, and raw Amenities string. Amenities_Clean is retained as a property attribute.")
st.markdown("#### Limitations")
st.markdown("The model estimates observed listing prices, not transaction value or future appreciation. A random split can overstate performance on unseen locations or future periods; there is no date-based holdout. Feature importance is model-specific and not causal. Evaluation metrics above come from the saved training run.")
