import streamlit as st
from src.analysis import apply_filters

def render_sidebar(df):
    st.sidebar.markdown("<div class='brand-mark'>PI</div><div class='brand-title'>Property Intelligence</div><div class='brand-sub'>INDIA · MARKET ANALYTICS</div><hr>",unsafe_allow_html=True)
    if st.sidebar.button("Reset all filters",use_container_width=True):
        for key in list(st.session_state):
            if key.startswith("flt_"): del st.session_state[key]
    f={}
    state=st.sidebar.multiselect("State",sorted(df.State.dropna().unique()),key="flt_state")
    f["State"]=state
    cities=sorted(df[df.State.isin(state)].City.dropna().unique()) if state else sorted(df.City.dropna().unique())
    f["City"]=st.sidebar.multiselect("City",cities,key="flt_city")
    for label,col in (("Property type","Property_Type"),("BHK","BHK"),("Furnishing","Furnished_Status"),("Availability","Availability_Status"),("Transport","Public_Transport_Accessibility"),("Parking","Parking_Space"),("Security","Security"),("Owner type","Owner_Type")):
        f[label]=st.sidebar.multiselect(label,sorted(df[col].dropna().unique()),key="flt_"+label.replace(" ","_").lower())
    pmin,pmax=float(df.Price_in_Lakhs.min()),float(df.Price_in_Lakhs.max())
    f["Price range"]=st.sidebar.slider("Price (₹ lakh)",pmin,pmax,(pmin,pmax),key="flt_price")
    smin,smax=int(df.Size_in_SqFt.min()),int(df.Size_in_SqFt.max())
    f["Size range"]=st.sidebar.slider("Size (sq ft)",smin,smax,(smin,smax),key="flt_size")
    out=apply_filters(df,f)
    st.sidebar.caption(f"{len(out):,} matching · {len(df):,} total")
    st.sidebar.markdown("<div class='sidebar-foot'>Analysis uses supplied listing data.<br>Not financial advice.</div>",unsafe_allow_html=True)
    return out
