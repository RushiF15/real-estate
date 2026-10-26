import pandas as pd

FILTERS={"State":"State","City":"City","Property type":"Property_Type","BHK":"BHK","Furnishing":"Furnished_Status","Availability":"Availability_Status","Transport":"Public_Transport_Accessibility","Parking":"Parking_Space","Security":"Security","Owner type":"Owner_Type"}

def apply_filters(df, filters):
    out=df
    for key,col in FILTERS.items():
        vals=filters.get(key)
        if vals and len(vals): out=out[out[col].isin(vals)]
    for key,col in (("Price range","Price_in_Lakhs"),("Size range","Size_in_SqFt")):
        bounds=filters.get(key)
        if bounds and len(bounds)==2: out=out[out[col].between(bounds[0],bounds[1])]
    return out

def overview_insights(df):
    if df.empty:return ["No rows are available for the current selection."]
    kind=df.Property_Type.value_counts().idxmax()
    state=df.groupby("State").Price_in_Lakhs.mean().sort_values(ascending=False)
    city=df.groupby("City").size().sort_values(ascending=False)
    return [f"**{kind}** is the most frequently listed property type ({df.Property_Type.value_counts().max():,} listings).",f"{state.index[0]} has the highest average listed price in this selection (₹{state.iloc[0]:,.1f} lakh).",f"Prices range from ₹{df.Price_in_Lakhs.min():,.1f} lakh to ₹{df.Price_in_Lakhs.max():,.1f} lakh; the median is ₹{df.Price_in_Lakhs.median():,.1f} lakh.",f"{city.index[0]} has the most listings ({city.iloc[0]:,}). These are descriptive listings, not verified transaction outcomes."]

def grouped_summary(df, group, value, agg="median", min_count=1):
    return df.groupby(group,dropna=False)[value].agg(agg).rename("value").to_frame().join(df.groupby(group,dropna=False).size().rename("count")).query("count >= @min_count").reset_index()

def correlation_data(df):
    cols=[c for c in ["BHK","Size_in_SqFt","Price_in_Lakhs","Price_per_SqFt_Clean","Year_Built","Age_of_Property","Nearby_Schools","Nearby_Hospitals","Floor_No","Total_Floors"] if c in df]
    return df[cols].corr(numeric_only=True)
