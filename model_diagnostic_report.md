# Current-Price Random Forest Diagnostic Report

**Date:** 2026-11-02  
**Dataset:** `india_housing_prices_cleaned.csv`  
**Model:** saved `models/current_price_pipeline.joblib`  
**Target:** `Price_in_Lakhs` (₹ lakh)

This report evaluates the existing saved model and its original 50,000-row random holdout. The holdout rows were reconstructed with the same 80/20 split and seed (42); their target values match the saved metrics exactly. The source CSV and model artifact were not changed.

## Summary

The Random Forest is effectively predicting close to the overall average price. Predictions have a standard deviation of **₹14.77 lakh**, compared with **₹141.19 lakh** for actual holdout prices. Their correlation is **−0.0043**, and the fitted prediction-versus-actual slope is approximately zero.

The Random Forest's holdout performance was slightly worse than the mean DummyRegressor on all three metrics:

| Model | MAE (₹ lakh) | RMSE (₹ lakh) | R² |
|---|---:|---:|---:|
| Random Forest | 122.8120 | 142.0259 | −0.011831 |
| DummyRegressor (mean) | 122.3178 | 141.1932 | −0.000001 |

The negative R² means the Random Forest did worse than predicting the training-set mean. This is a dataset-level result; it does not support a reliable real-estate price predictor.

## 1. Price relationships

Pearson and Spearman correlations were computed on all numeric columns. The two rate columns correlate with price, but both must remain excluded: `Price_per_SqFt_Clean` is calculated directly from price and area, while the original `Price_per_SqFt` has an invalid 0–0.99 scale and unclear provenance.

| Numeric feature | Pearson r | Spearman r | Model input? |
|---|---:|---:|---|
| `Price_per_SqFt_Clean` | 0.55578 | 0.75154 | No; derived from target |
| `Price_per_SqFt` | 0.55562 | 0.75023 | No; invalid scale/unclear origin |
| `Nearby_Hospitals` | −0.00277 | −0.00277 | Yes |
| `Year_Built` | 0.00271 | 0.00272 | Yes |
| `Age_of_Property` | −0.00271 | −0.00272 | Yes |
| `Size_in_SqFt` | −0.00253 | −0.00253 | Yes |
| `Floor_No` | −0.00172 | −0.00173 | Yes |
| `ID` | −0.00164 | −0.00163 | No |
| `Total_Floors` | 0.00128 | 0.00129 | Yes |
| `BHK` | −0.00098 | −0.00099 | Yes |
| `Nearby_Schools` | 0.00015 | 0.00015 | Yes |

Ignoring the target and derived rate fields, **every numeric predictor has an absolute correlation below 0.003** with price. This includes size, BHK, year, age, floors, nearby schools and nearby hospitals.

Categorical predictors do have multiple levels, but the price differences between those levels are very small. The table below uses η² (eta-squared), the fraction of price variation associated with the groups for each category. These values are descriptive and do not imply causation.

| Categorical field | Levels | η² |
|---|---:|---:|
| `Locality` | 500 | 0.002041 |
| `Amenities` (raw; not a model input) | 325 | 0.001101 |
| `City` | 42 | 0.000190 |
| `Amenities_Clean` | 31 | 0.000099 |
| `State` | 20 | 0.000097 |
| `Property_Type` | 3 | 0.000022 |
| `Security` | 2 | 0.000015 |
| `Public_Transport_Accessibility` | 3 | 0.000009 |
| `Availability_Status` | 2 | 0.000008 |
| `Facing` | 4 | 0.000006 |
| `Furnished_Status` | 3 | 0.000004 |
| `Owner_Type` | 3 | 0.000004 |
| `Parking_Space` | 2 | 0.000001 |

For context, city mean prices span only **₹250.79–₹258.46 lakh**, against an overall target standard deviation of **₹141.35 lakh**. Property-type means span **₹253.77–₹255.37 lakh**; BHK means span **₹254.09–₹254.88 lakh**. The groups are not empty or missing, but they carry almost no price separation in this dataset.

## 2. Category variation

Categorical fields have broad coverage and generally balanced counts:

| Field | Levels | Count per level (min–max) |
|---|---:|---:|
| `State` | 20 | 12,314–12,681 |
| `City` | 42 | 4,063–6,461 |
| `Property_Type` | 3 | 82,956–83,744 |
| `BHK` | 5 | About 50,000 each |
| `Furnished_Status` | 3 | 83,218–83,408 |
| `Public_Transport_Accessibility` | 3 | 83,008–83,705 |
| `Parking_Space` | 2 | 124,544–125,456 |
| `Security` | 2 | 124,767–125,233 |
| `Facing` | 4 | 62,269–62,757 |
| `Owner_Type` | 3 | 83,253–83,479 |
| `Availability_Status` | 2 | 124,965–125,035 |
| `Amenities_Clean` | 31 | 4,898–49,864 |

The counts show variation, but variation by itself does not make a feature predictive. In this file, the categories are close to evenly represented while their price means are nearly identical. `Locality` has 500 generic labels reused across cities, so it is not a unique geographic identifier.

## 3. Random Forest feature importance

Feature importances below are averaged from the saved forest's trees. One-hot categorical importances have been summed back to their original field. Importance is relative to this fitted model; it is not evidence of a real price effect or of good test performance.

| Original feature | Importance |
|---|---:|
| `Size_in_SqFt` | 0.1514 |
| `Total_Floors` | 0.0889 |
| `Floor_No` | 0.0880 |
| `Amenities_Clean` | 0.0642 |
| `Nearby_Schools` | 0.0634 |
| `Nearby_Hospitals` | 0.0609 |
| `Year_Built` | 0.0526 |
| `City` | 0.0522 |
| `Age_of_Property` | 0.0522 |
| `State` | 0.0469 |
| `Facing` | 0.0440 |
| `BHK` | 0.0428 |
| `Furnished_Status` | 0.0353 |
| `Public_Transport_Accessibility` | 0.0352 |
| `Owner_Type` | 0.0350 |
| `Property_Type` | 0.0281 |
| `Parking_Space` | 0.0202 |
| `Security` | 0.0194 |
| `Availability_Status` | 0.0192 |

The forest assigns importance to many unrelated fields even though their relationships with price are negligible and the holdout score is worse than the mean baseline. Here, importance reflects how the trees split this dataset, including splits on noise; it should not be read as proof that floors, schools or amenities explain actual market prices.

## 4. Actual prices versus predictions

| Holdout statistic | Actual price (₹ lakh) | Prediction (₹ lakh) |
|---|---:|---:|
| Mean | 254.715 | 254.588 |
| Standard deviation | 141.193 | 14.766 |
| Minimum | 10.020 | 191.579 |
| 25th percentile | 132.920 | 244.647 |
| Median | 252.825 | 254.690 |
| 75th percentile | 377.403 | 264.592 |
| Maximum | 499.960 | 317.411 |

The forest produced 50,000 distinct numerical predictions, so it is not literally returning one fixed number. But those predictions occupy a narrow band of about **₹126 lakh**, while the actual prices span about **₹490 lakh**. Its correlation with actual price is **−0.0043**. This is near-constant, mean-like prediction in practical terms.

Absolute error was **₹122.81 lakh on average** and **₹122.31 lakh at the median**. The 90th-percentile absolute error was **₹220.90 lakh**. Only 10.1% of predictions were within ₹25 lakh, 20.2% within ₹50 lakh, and 40.8% within ₹100 lakh of the actual price.

## 5. Error by group

All group metrics use the same 50,000-row holdout. Bias is predicted minus actual price; a negative value indicates underprediction.

| Grouping | Groups | MAE range (₹ lakh) | RMSE range (₹ lakh) | Bias range (₹ lakh) |
|---|---:|---:|---:|---:|
| City | 42 | 118.16–129.18 | 137.17–147.05 | −9.90 to 7.23 |
| Property type | 3 | 122.17–123.55 | 141.41–142.67 | −1.26 to 0.96 |
| BHK | 5 | 121.74–123.51 | 141.21–142.69 | −1.88 to 2.73 |

The city with the lowest MAE was Hyderabad (**₹118.16 lakh**, 1,320 rows); the highest was Bangalore (**₹129.18 lakh**, 833 rows). Differences by property type and BHK are small because their prices and predictions are all centered near ₹255 lakh. No group shows good accuracy.

## 6. Data issues and likely reasons for failure

- **Price appears nearly uniform.** The target ranges from ₹10–₹500 lakh, and ten equal-width bins contain 24,725–25,465 records each. Every price is rounded to cents (two decimal places), with 48,700 distinct values. This unusually even distribution may indicate generated or otherwise assigned values. It is a warning sign, not proof of synthetic origin; dataset provenance remains unknown.
- **Attributes scarcely track the target.** Size-price correlation is −0.00253, city η² is 0.000190, and the categorical group means are nearly flat. The available predictors do not explain the observed price spread.
- **Major floor inconsistencies remain.** `Floor_No > Total_Floors` in 116,304 records (46.52%); `Floor_No` is zero in 8,203 records (3.28%). These can make floor-related feature importance especially hard to interpret.
- **Age duplicates year.** `Age_of_Property = 2025 − Year_Built` in every row. Including both gives the model redundant inputs.
- **Derived price rates must stay excluded.** `Price_per_SqFt_Clean` exactly equals `Price_in_Lakhs × 100,000 / Size_in_SqFt`, so its high price correlation is direct target leakage. The fitted model correctly excluded both rate columns.
- **Generic locality IDs have limited geographic meaning.** The 500 `Locality` labels are reused across cities and are omitted from the existing model. They cannot identify a real neighborhood without a city-qualified mapping.
- **No transaction dates or verified sale outcomes are present.** The fitted target is the supplied price field, whose real-world meaning and source have not been verified. A random split tests only interpolation within this file, not future or market-wide performance.

The file has no missing cells or exact duplicate rows, and its numeric values are finite. Those basic checks do not resolve the weak target-feature relationships or the provenance concerns.

## Conclusion

The diagnostic evidence explains the failed baseline comparison: the target has a wide spread, but the predictors and categories contain almost no corresponding price signal. The forest compresses predictions toward the mean, with nearly zero actual/predicted correlation, and loses to a simple mean predictor. Until the target's origin and relationship to property attributes can be verified—or a better sourced dataset is used—this model should not be presented as a meaningful property valuation tool.
