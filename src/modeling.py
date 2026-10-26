from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET="Price_in_Lakhs"
LEAKAGE_COLUMNS={"ID","Price_in_Lakhs","Price_per_SqFt","Price_per_SqFt_Clean","Amenities"}
NUMERIC_FEATURES=["BHK","Size_in_SqFt","Year_Built","Age_of_Property","Floor_No","Total_Floors","Nearby_Schools","Nearby_Hospitals"]
CATEGORICAL_FEATURES=["State","City","Property_Type","Furnished_Status","Public_Transport_Accessibility","Parking_Space","Security","Facing","Owner_Type","Availability_Status","Amenities_Clean"]
FEATURES=NUMERIC_FEATURES+CATEGORICAL_FEATURES
MODEL_PATH=Path("models/current_price_pipeline.joblib")

def make_pipeline():
    numeric=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    categorical=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=2))])
    prep=ColumnTransformer([("numeric",numeric,NUMERIC_FEATURES),("categorical",categorical,CATEGORICAL_FEATURES)])
    return Pipeline([("preprocess",prep),("regressor",RandomForestRegressor(n_estimators=120,min_samples_leaf=3,max_features=.8,n_jobs=-1,random_state=42))])

def train_model(df,model_path=MODEL_PATH):
    features=[c for c in FEATURES if c in df.columns]
    if len(df)<40: raise ValueError("At least 40 complete observations are required for a train/test split.")
    X=df[features].copy(); y=df[TARGET].astype(float)
    if y.isna().any(): raise ValueError("Target contains missing prices.")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    model=make_pipeline(); model.set_params(preprocess__transformers=[("numeric",model.named_steps["preprocess"].transformers[0][1],[c for c in NUMERIC_FEATURES if c in features]),("categorical",model.named_steps["preprocess"].transformers[1][1],[c for c in CATEGORICAL_FEATURES if c in features])])
    model.fit(X_train,y_train); pred=model.predict(X_test)
    metrics={"mae":float(mean_absolute_error(y_test,pred)),"rmse":float(np.sqrt(mean_squared_error(y_test,pred))),"r2":float(r2_score(y_test,pred)),"rows":len(df),"train_rows":len(X_train),"test_rows":len(X_test),"test_size":.2,"features":features,"y_test":y_test.to_numpy(),"predictions":pred}
    Path(model_path).parent.mkdir(parents=True,exist_ok=True); joblib.dump({"model":model,"metrics":metrics,"features":features,"target":TARGET},model_path)
    return model,metrics

def load_model(model_path=MODEL_PATH):
    p=Path(model_path)
    return joblib.load(p) if p.exists() else None

def leakage_check(features=FEATURES):
    return sorted(set(features)&LEAKAGE_COLUMNS)

def feature_importance(bundle):
    model=bundle["model"]; prep=model.named_steps["preprocess"]
    names=prep.get_feature_names_out(); values=model.named_steps["regressor"].feature_importances_
    return pd.DataFrame({"Feature":names,"Importance":values}).sort_values("Importance",ascending=False)
