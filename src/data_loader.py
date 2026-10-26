import os
from pathlib import Path
import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = ["ID","State","City","Locality","Property_Type","BHK","Size_in_SqFt","Price_in_Lakhs","Year_Built","Furnished_Status","Floor_No","Total_Floors","Age_of_Property","Nearby_Schools","Nearby_Hospitals","Public_Transport_Accessibility","Parking_Space","Security","Amenities","Facing","Owner_Type","Availability_Status"]

def resolve_data_path():
    configured=os.getenv("HOUSING_DATA_PATH")
    candidates=[Path(configured)] if configured else []
    candidates += [Path("data/india_housing_prices_cleaned.csv"),Path("data/india_housing_prices_cleaned (1).csv"),Path("india_housing_prices_cleaned.csv"),Path("india_housing_prices_cleaned (1).csv")]
    return next((p for p in candidates if p.exists()),None)

@st.cache_data(show_spinner="Loading housing data…")
def load_data(path=None):
    file_path=Path(path) if path else resolve_data_path()
    if file_path is None or not file_path.exists(): return None
    df=pd.read_csv(file_path,low_memory=False)
    missing=set(REQUIRED_COLUMNS)-set(df.columns)
    if missing: raise ValueError("Dataset is missing required columns: "+", ".join(sorted(missing)))
    if "Price_per_SqFt_Clean" not in df:
        df["Price_per_SqFt_Clean"]=(df["Price_in_Lakhs"]*100000/df["Size_in_SqFt"].replace(0,pd.NA))
    return df

def validate_data(df):
    age=(df["Age_of_Property"] != 2025-df["Year_Built"]) if {"Age_of_Property","Year_Built"}<=set(df) else pd.Series(dtype=bool)
    bad=(df["Floor_No"]>df["Total_Floors"]) if {"Floor_No","Total_Floors"}<=set(df) else pd.Series(dtype=bool)
    return {"missing_cells":int(df.isna().sum().sum()),"duplicate_rows":int(df.duplicated().sum()),"age_mismatch":int(age.sum()),"bad_floors":int(bad.sum())}
