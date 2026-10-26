import streamlit as st
import pandas as pd
from components.styling import apply_style
from src.data_loader import load_data
from src.sidebar import render_sidebar

apply_style(); st.title("Investment advisor"); st.caption("Compare a property profile with similar listings and an exploratory score. This is not a validated success probability or financial advice.")
df=load_data()
if df is None: st.error("Dataset missing. Configure HOUSING_DATA_PATH."); st.stop()
d=render_sidebar(df)
if d.empty: st.info("No properties match global filters."); st.stop()
with st.form("profile"):
    st.subheader("Build a property profile")
    a,b,c=st.columns(3)
    states=sorted(d.State.unique()); state=a.selectbox("State",states); cities=sorted(d[d.State==state].City.unique()); city=b.selectbox("City",cities)
    ptype=c.selectbox("Property type",sorted(d.Property_Type.unique())); bhk=a.selectbox("BHK",sorted(d.BHK.unique())); size=b.number_input("Size (sq ft)",min_value=100,max_value=10000,value=int(d.Size_in_SqFt.median()),step=50)
    furnished=c.selectbox("Furnishing",sorted(d.Furnished_Status.unique())); year=a.number_input("Year built",min_value=1900,max_value=2026,value=2015); schools=b.number_input("Nearby schools",min_value=0,max_value=30,value=5); hospitals=c.number_input("Nearby hospitals",min_value=0,max_value=30,value=5)
    transport=a.selectbox("Transport access",sorted(d.Public_Transport_Accessibility.unique())); parking=b.selectbox("Parking",sorted(d.Parking_Space.unique())); security=c.selectbox("Security",sorted(d.Security.unique()))
    facing=a.selectbox("Facing",sorted(d.Facing.unique())); owner=b.selectbox("Owner type",sorted(d.Owner_Type.unique())); availability=c.selectbox("Availability",sorted(d.Availability_Status.unique()))
    amenities=sorted(set(x.strip() for row in d.Amenities_Clean.dropna().head(25000) for x in row.split(","))); selected_amenities=st.multiselect("Amenities",amenities)
    submitted=st.form_submit_button("Assess profile",type="primary")
if submitted:
    peer=d[(d.City==city)&(d.Property_Type==ptype)&(d.BHK==bhk)].copy()
    if peer.empty: peer=d[(d.City==city)&(d.Property_Type==ptype)].copy()
    if peer.empty: peer=d[(d.Property_Type==ptype)&(d.BHK==bhk)].copy()
    if peer.empty: peer=d.copy()
    target_rate=peer.Price_per_SqFt_Clean.median(); target_size=(peer.Size_in_SqFt-size).abs().div(max(size,1)); target_age=(peer.Year_Built-year).abs()/33
    peers=peer.assign(_distance=target_size+target_age).nsmallest(25,"_distance")
    amenity_match=peer.Amenities_Clean.fillna("").map(lambda x: all(a in [v.strip() for v in x.split(",")] for a in selected_amenities))
    score_parts={"Price efficiency":max(0,min(100,100*(1-abs((size*target_rate/100000)-peer.Price_in_Lakhs.median())/max(peer.Price_in_Lakhs.median(),1)))),"Schools":float((peer.Nearby_Schools>=schools).mean()*100),"Hospitals":float((peer.Nearby_Hospitals>=hospitals).mean()*100),"Transport":float((peer.Public_Transport_Accessibility==transport).mean()*100),"Parking":float((peer.Parking_Space==parking).mean()*100),"Security":float((peer.Security==security).mean()*100),"Furnishing":float((peer.Furnished_Status==furnished).mean()*100),"Facing":float((peer.Facing==facing).mean()*100),"Owner type":float((peer.Owner_Type==owner).mean()*100),"Availability":float((peer.Availability_Status==availability).mean()*100),"Selected amenities":float(amenity_match.mean()*100)}
    score=sum(score_parts.values())/len(score_parts)
    st.markdown("### Profile summary")
    st.dataframe(pd.DataFrame({"Profile feature":["State","City","Property type","BHK","Size","Built","Furnishing","Nearby schools / hospitals","Transport","Parking / security","Facing","Owner type","Availability","Amenities"],"Selected value":[state,city,ptype,bhk,f"{size:,} sq ft",year,furnished,f"{schools} / {hospitals}",transport,f"{parking} / {security}",facing,owner,availability,", ".join(selected_amenities) or "No amenity requirement"]}),use_container_width=True,hide_index=True)
    cols=st.columns(4); cols[0].metric("Peer listings",f"{len(peer):,}"); cols[1].metric("Peer median price",f"₹{peer.Price_in_Lakhs.median():,.1f}L"); cols[2].metric("Peer median rate",f"₹{target_rate:,.0f}/sq ft"); cols[3].metric("Exploratory fit score",f"{score:.0f}/100")
    st.caption("Exploratory fit score: equally weighted average of 11 peer-relative checks. Price efficiency compares the profile's estimated price (profile area × peer median corrected rate) with peer median price; the remaining checks are the share of peer listings meeting the selected profile criteria. This score is not trained, validated, or a chance of investment success.")
    st.dataframe(pd.DataFrame({"Comparison":score_parts.keys(),"Peer-relative score (0–100)":score_parts.values()}),use_container_width=True,hide_index=True)
    st.markdown("#### Closest comparable listings")
    st.dataframe(peers[["State","City","Property_Type","BHK","Size_in_SqFt","Price_in_Lakhs","Price_per_SqFt_Clean","Year_Built","Furnished_Status","Availability_Status"]],use_container_width=True,hide_index=True)
    st.info("Listing comparisons do not establish rental yield, capital appreciation, liquidity, legal title, or investment returns. Transaction and rental data would be needed for those assessments.")
else: st.info("Enter a profile and select **Assess profile** to compare it with listings matching the active dashboard filters.")
