from src.data_loader import load_data
from src.modeling import train_model
# it is video based ppt so dont add things like ask questions etc and 

if __name__ == "__main__":
    data=load_data()
    if data is None: raise SystemExit("Dataset not found. Place CSV in data/ or set HOUSING_DATA_PATH.")
    _,metrics=train_model(data)
    print("Saved model: models/current_price_pipeline.joblib")
    print(f"Rows: {metrics['valid_rows']:,} valid / {metrics['excluded_rows']:,} excluded / {metrics['total_rows']:,} total")
    print(f"Split: {metrics['split']} (random_state={metrics['random_state']})")
    print("Metrics on the same held-out test rows (price in lakhs):")
    print("Random Forest: "
          f"MAE={metrics['random_forest']['mae']:.4f} | "
          f"RMSE={metrics['random_forest']['rmse']:.4f} | "
          f"R2={metrics['random_forest']['r2']:.6f}")
    print("DummyRegressor (mean): "
          f"MAE={metrics['dummy_regressor']['mae']:.4f} | "
          f"RMSE={metrics['dummy_regressor']['rmse']:.4f} | "
          f"R2={metrics['dummy_regressor']['r2']:.6f}")
    better=(metrics['random_forest']['mae']<metrics['dummy_regressor']['mae']
            and metrics['random_forest']['rmse']<metrics['dummy_regressor']['rmse']
            and metrics['random_forest']['r2']>metrics['dummy_regressor']['r2'])
    print(f"Random Forest better than baseline on all three metrics: {'Yes' if better else 'No'}")
    print("This is a random holdout evaluation of the supplied dataset; it does not establish real-world reliability.")
