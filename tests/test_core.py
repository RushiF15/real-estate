import numpy as np
import pandas as pd
import pytest
from src.analysis import apply_filters,overview_insights,grouped_summary
from src.data_loader import REQUIRED_COLUMNS,load_data,validate_data
from src.modeling import FEATURES,leakage_check,train_model,load_model

@pytest.fixture
def frame():
    n=60
    return pd.DataFrame({"ID":range(n),"State":["A","B"]*(n//2),"City":["X","Y"]*(n//2),"Locality":["L"]*n,"Property_Type":["Apartment"]*n,"BHK":[2]*n,"Size_in_SqFt":np.arange(500,560),"Price_in_Lakhs":np.arange(100,160,dtype=float),"Price_per_SqFt":np.zeros(n),"Price_per_SqFt_Clean":np.arange(100,160)*100000/np.arange(500,560),"Year_Built":[2010]*n,"Furnished_Status":["Furnished"]*n,"Floor_No":[1]*n,"Total_Floors":[5]*n,"Age_of_Property":[15]*n,"Nearby_Schools":[3]*n,"Nearby_Hospitals":[2]*n,"Public_Transport_Accessibility":["High"]*n,"Parking_Space":["Yes"]*n,"Security":["Yes"]*n,"Amenities":["Gym"]*n,"Amenities_Clean":["Gym"]*n,"Facing":["East"]*n,"Owner_Type":["Owner"]*n,"Availability_Status":["Ready_to_Move"]*n})

def test_required_columns_and_validation(frame):
    assert set(REQUIRED_COLUMNS)<=set(frame)
    assert validate_data(frame)=={"missing_cells":0,"duplicate_rows":0,"age_mismatch":0,"bad_floors":0}

def test_filters_and_empty(frame):
    assert len(apply_filters(frame,{"State":["A"],"Price range":(100,110)}))==6
    assert apply_filters(frame,{"State":["missing"]}).empty
    assert len(apply_filters(frame,{}))==len(frame)

def test_derived_statistics(frame):
    assert overview_insights(frame)
    assert grouped_summary(frame,"State","Price_in_Lakhs").shape[0]==2

def test_leakage_prevention():
    assert not set(leakage_check(FEATURES))
    assert "Price_per_SqFt_Clean" in leakage_check(["Price_per_SqFt_Clean"])

def test_training_and_missing_model(frame,tmp_path):
    path=tmp_path/"pipeline.joblib"
    assert load_model(path) is None
    model,metrics=train_model(frame,path)
    assert path.exists() and metrics["test_rows"]==12
    assert np.isfinite(model.predict(frame[metrics["features"]])).all()

def test_loader_reports_schema_error(tmp_path):
    p=tmp_path/"bad.csv"; pd.DataFrame({"oops":[1]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match="missing required columns"): load_data(str(p))
