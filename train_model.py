from src.data_loader import load_data
from src.modeling import train_model

if __name__ == "__main__":
    data=load_data()
    if data is None: raise SystemExit("Dataset not found. Place CSV in data/ or set HOUSING_DATA_PATH.")
    _,metrics=train_model(data)
    print(f"Saved models/current_price_pipeline.joblib using {metrics['rows']:,} rows")
    print(f"Holdout MAE: ₹{metrics['mae']:.2f} lakh | RMSE: ₹{metrics['rmse']:.2f} lakh | R²: {metrics['r2']:.4f}")
