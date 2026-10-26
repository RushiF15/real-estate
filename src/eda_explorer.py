"""Structured, data-backed question definitions and answer builders for the EDA explorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from components.charts import apply_chart_theme
from src.analysis import correlation_data


def _sub(label, kind, **kwargs):
    return {"label": label, "kind": kind, **kwargs}


QUESTION_CATEGORIES = [
    {
        "category": "Property Prices and Sizes",
        "questions": [
            {"id": "q1", "title": "Property Price Distribution", "subquestions": [
                _sub("What is the overall distribution of property prices?", "distribution", field="Price_in_Lakhs"),
                _sub("What is the average property price?", "scalar", field="Price_in_Lakhs", agg="mean"),
                _sub("What is the median property price?", "scalar", field="Price_in_Lakhs", agg="median"),
                _sub("Which price range contains the most properties?", "common_bin", field="Price_in_Lakhs", bins=10),
                _sub("How does the price distribution compare with the full dataset?", "filtered_compare", field="Price_in_Lakhs"),
            ]},
            {"id": "q2", "title": "Property Size Distribution", "subquestions": [
                _sub("What is the distribution of property sizes?", "distribution", field="Size_in_SqFt"),
                _sub("What are the average and median property sizes?", "summary", field="Size_in_SqFt"),
                _sub("Which size range is most common?", "common_bin", field="Size_in_SqFt", bins=10),
                _sub("How does property size vary by property type?", "group", field="Size_in_SqFt", group="Property_Type", agg="median"),
            ]},
            {"id": "q3", "title": "Price per Square Foot", "subquestions": [
                _sub("Which property type has the highest average corrected price per square foot?", "group", field="Price_per_SqFt_Clean", group="Property_Type", agg="mean", rank="highest"),
                _sub("How does corrected price per square foot vary by property type?", "group_box", field="Price_per_SqFt_Clean", group="Property_Type"),
                _sub("How does corrected price per square foot vary by state?", "group", field="Price_per_SqFt_Clean", group="State", agg="median", top=15),
                _sub("Which cities have the highest median corrected price per square foot?", "group", field="Price_per_SqFt_Clean", group="City", agg="median", top=15, min_count=10),
            ]},
            {"id": "q4", "title": "Property Size and Price Relationship", "subquestions": [
                _sub("What is the relationship between property size and price?", "scatter", x="Size_in_SqFt", y="Price_in_Lakhs"),
                _sub("How strong is the linear association between size and price?", "pair_correlation", x="Size_in_SqFt", y="Price_in_Lakhs"),
                _sub("How does median price change across property-size bands?", "binned_group", field="Price_in_Lakhs", by="Size_in_SqFt", bins=10),
                _sub("How does average size vary across price bands?", "binned_group", field="Size_in_SqFt", by="Price_in_Lakhs", bins=10),
            ]},
            {"id": "q5", "title": "Potential Outliers", "subquestions": [
                _sub("How many potential outliers are present in corrected price per square foot?", "outlier", field="Price_per_SqFt_Clean"),
                _sub("How many potential outliers are present in property size?", "outlier", field="Size_in_SqFt"),
                _sub("How do potential outlier counts compare across these measures?", "outlier_compare"),
                _sub("What do the price-per-square-foot and size box plots look like?", "outlier_box_both"),
            ]},
        ],
    },
    {
        "category": "Location and Market Comparison",
        "questions": [
            {"id": "q6", "title": "Corrected Price per Square Foot by State", "subquestions": [
                _sub("Which states have the highest average corrected price per square foot?", "group", field="Price_per_SqFt_Clean", group="State", agg="mean", top=15, rank="highest"),
                _sub("How do state median corrected rates compare?", "group", field="Price_per_SqFt_Clean", group="State", agg="median", top=15),
                _sub("Which state has the lowest average corrected rate?", "group", field="Price_per_SqFt_Clean", group="State", agg="mean", top=15, rank="lowest"),
                _sub("How many properties contribute to each state comparison?", "group", field="ID", group="State", agg="count", top=20),
            ]},
            {"id": "q7", "title": "Property Prices by City", "subquestions": [
                _sub("Which cities have the highest average property prices?", "group", field="Price_in_Lakhs", group="City", agg="mean", top=15, min_count=10, rank="highest"),
                _sub("How do city median property prices compare?", "group", field="Price_in_Lakhs", group="City", agg="median", top=15, min_count=10),
                _sub("Which cities have the most listings?", "group", field="ID", group="City", agg="count", top=15),
                _sub("How far apart are the highest and lowest city averages?", "group", field="Price_in_Lakhs", group="City", agg="mean", top=15, min_count=10, rank="spread"),
            ]},
            {"id": "q8", "title": "Property Age by Locality", "subquestions": [
                _sub("Which localities have the highest median property age?", "group", field="Age_of_Property", group="Locality", agg="median", top=20, rank="highest"),
                _sub("Which localities have the lowest median property age?", "group", field="Age_of_Property", group="Locality", agg="median", top=20, rank="lowest"),
                _sub("How is property age distributed across the selected data?", "distribution", field="Age_of_Property"),
                _sub("How does property age relate to listing price?", "scatter", x="Age_of_Property", y="Price_in_Lakhs"),
            ]},
            {"id": "q9", "title": "BHK Distribution Across Cities", "subquestions": [
                _sub("What is the overall distribution of BHK configurations?", "distribution", field="BHK", discrete=True),
                _sub("How are BHK configurations distributed across cities?", "city_bhk_heatmap"),
                _sub("What is the most common BHK configuration in each city?", "city_bhk_mode"),
                _sub("How does median property price vary by BHK?", "group", field="Price_in_Lakhs", group="BHK", agg="median"),
            ]},
            {"id": "q10", "title": "Five Most Expensive Localities", "subquestions": [
                _sub("Which five localities have the highest average listing prices?", "top_localities"),
                _sub("How do median prices compare in those five localities?", "top_locality_medians"),
                _sub("How does price vary by construction year in those localities?", "locality_year_trend"),
                _sub("How many selected listings are available in each top locality?", "top_locality_counts"),
            ]},
        ],
    },
    {
        "category": "Property Characteristics",
        "questions": [
            {"id": "q11", "title": "Numeric Variable Correlations", "subquestions": [
                _sub("Which numeric variables are correlated?", "correlation_heatmap"),
                _sub("What is the correlation between size and price?", "pair_correlation", x="Size_in_SqFt", y="Price_in_Lakhs"),
                _sub("How are nearby schools and hospitals associated?", "pair_correlation", x="Nearby_Schools", y="Nearby_Hospitals"),
                _sub("Which numeric pair has the strongest absolute correlation?", "strongest_correlation"),
            ]},
            {"id": "q14", "title": "Price by Furnishing Status", "subquestions": [
                _sub("How does median property price vary by furnishing status?", "group", field="Price_in_Lakhs", group="Furnished_Status", agg="median"),
                _sub("How does average property price vary by furnishing status?", "group", field="Price_in_Lakhs", group="Furnished_Status", agg="mean"),
                _sub("How does the price distribution differ by furnishing status?", "group_box", field="Price_in_Lakhs", group="Furnished_Status"),
                _sub("How does property size vary by furnishing status?", "group", field="Size_in_SqFt", group="Furnished_Status", agg="median"),
            ]},
            {"id": "q15", "title": "Corrected Price per Square Foot by Facing", "subquestions": [
                _sub("Which facing direction has the highest median corrected rate?", "group", field="Price_per_SqFt_Clean", group="Facing", agg="median", rank="highest"),
                _sub("How does average corrected rate vary by facing direction?", "group", field="Price_per_SqFt_Clean", group="Facing", agg="mean"),
                _sub("How many listings are represented in each facing group?", "group", field="ID", group="Facing", agg="count"),
                _sub("How do corrected-rate distributions vary by facing direction?", "group_box", field="Price_per_SqFt_Clean", group="Facing"),
            ]},
            {"id": "q16", "title": "Property Counts by Owner Type", "subquestions": [
                _sub("How many properties are listed by each owner type?", "group", field="ID", group="Owner_Type", agg="count"),
                _sub("What share of listings belongs to each owner type?", "share", group="Owner_Type"),
                _sub("How does median property price vary by owner type?", "group", field="Price_in_Lakhs", group="Owner_Type", agg="median"),
                _sub("How does average corrected rate vary by owner type?", "group", field="Price_per_SqFt_Clean", group="Owner_Type", agg="mean"),
            ]},
            {"id": "q17", "title": "Property Counts by Availability", "subquestions": [
                _sub("How many properties have each availability status?", "group", field="ID", group="Availability_Status", agg="count"),
                _sub("What share of listings has each availability status?", "share", group="Availability_Status"),
                _sub("How does median price vary by availability status?", "group", field="Price_in_Lakhs", group="Availability_Status", agg="median"),
                _sub("How does corrected rate vary by availability status?", "group", field="Price_per_SqFt_Clean", group="Availability_Status", agg="median"),
            ]},
            {"id": "q18", "title": "Parking and Property Price", "subquestions": [
                _sub("How many listings report parking availability?", "group", field="ID", group="Parking_Space", agg="count"),
                _sub("How does median price compare by parking availability?", "group", field="Price_in_Lakhs", group="Parking_Space", agg="median"),
                _sub("How does the price distribution vary by parking availability?", "group_box", field="Price_in_Lakhs", group="Parking_Space"),
                _sub("How does price vary jointly by parking and security?", "parking_security_heatmap"),
            ]},
        ],
    },
    {
        "category": "Amenities and Accessibility",
        "questions": [
            {"id": "q12", "title": "Nearby Schools and Corrected Price per Square Foot", "subquestions": [
                _sub("How are nearby school counts associated with corrected price per square foot?", "facility_association", field="Nearby_Schools"),
                _sub("What is the median corrected rate for each nearby-school count?", "facility_group", field="Nearby_Schools"),
                _sub("How many listings are available at each nearby-school count?", "facility_counts", field="Nearby_Schools"),
                _sub("How does listing price relate to nearby-school count?", "scatter", x="Nearby_Schools", y="Price_in_Lakhs"),
            ]},
            {"id": "q13", "title": "Nearby Hospitals and Corrected Price per Square Foot", "subquestions": [
                _sub("How are nearby hospital counts associated with corrected price per square foot?", "facility_association", field="Nearby_Hospitals"),
                _sub("What is the median corrected rate for each nearby-hospital count?", "facility_group", field="Nearby_Hospitals"),
                _sub("How many listings are available at each nearby-hospital count?", "facility_counts", field="Nearby_Hospitals"),
                _sub("How does listing price relate to nearby-hospital count?", "scatter", x="Nearby_Hospitals", y="Price_in_Lakhs"),
            ]},
            {"id": "q19", "title": "Amenities and Corrected Price per Square Foot", "subquestions": [
                _sub("Which amenity combinations have the highest median corrected rate?", "amenities", agg="median", rank="highest"),
                _sub("Which amenity combinations have the highest average corrected rate?", "amenities", agg="mean", rank="highest"),
                _sub("Which amenity combinations are most common?", "amenity_counts"),
                _sub("How does corrected rate vary across amenity combinations?", "amenities_box"),
            ]},
        ],
    },
    {
        "category": "Investment Analysis",
        "questions": [
            {"id": "q20", "title": "Transport Accessibility and Investment Context", "subquestions": [
                _sub("How does median listing price vary by transport accessibility?", "group", field="Price_in_Lakhs", group="Public_Transport_Accessibility", agg="median"),
                _sub("How does corrected price per square foot vary by accessibility?", "group", field="Price_per_SqFt_Clean", group="Public_Transport_Accessibility", agg="median"),
                _sub("How many properties are represented at each accessibility level?", "group", field="ID", group="Public_Transport_Accessibility", agg="count"),
                _sub("Which accessibility group has the highest median listing price?", "group", field="Price_in_Lakhs", group="Public_Transport_Accessibility", agg="median", rank="highest"),
                _sub("What investment conclusions can this accessibility comparison support?", "investment_limit"),
            ]},
        ],
    },
]

QUESTIONS = {q["id"]: q for category in QUESTION_CATEGORIES for q in category["questions"]}

FIELD_LABELS = {
    "Price_in_Lakhs": "Price (₹ lakh)",
    "Price_per_SqFt_Clean": "Corrected price per sq ft (₹)",
    "Size_in_SqFt": "Size (sq ft)",
    "Age_of_Property": "Age (years; 2025 reference)",
    "BHK": "BHK configuration",
    "Nearby_Schools": "Nearby schools",
    "Nearby_Hospitals": "Nearby hospitals",
    "Year_Built": "Year built",
    "ID": "Property count",
}


def _format(value, field):
    if field == "Price_in_Lakhs":
        return f"₹{value:,.2f} lakh"
    if field == "Price_per_SqFt_Clean":
        return f"₹{value:,.0f} per sq ft"
    if field == "Size_in_SqFt":
        return f"{value:,.0f} sq ft"
    if field == "Age_of_Property":
        return f"{value:,.1f} years"
    if field == "ID":
        return f"{value:,.0f} properties"
    return f"{value:,.2f}" if isinstance(value, (float, np.floating)) else str(value)


def _metric(df, field, agg):
    if agg == "count":
        return len(df)
    return df[field].agg(agg)


def _outlier_stats(series):
    q1, q3 = series.quantile([.25, .75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < low) | (series > high)).sum()), float(low), float(high)


def build_answer(df, full_df, spec):
    """Return answer text, chart, statistics, chart data and caveats for one subquestion."""
    kind = spec["kind"]
    empty = {"answer": "There are no records to analyze for this selection.", "stats": {}, "figure": None,
             "table": pd.DataFrame(), "interpretation": "Adjust the global filters to include properties.", "caveat": ""}
    if df.empty:
        return empty
    chart_data = pd.DataFrame()
    stats = {"Filtered properties": f"{len(df):,}"}
    answer = interpretation = caveat = ""
    fig = None
    figures = []
    field = spec.get("field")

    if kind in {"distribution", "scalar", "summary", "common_bin", "filtered_compare"}:
        series = df[field].dropna()
        label = FIELD_LABELS.get(field, field)
        if kind == "distribution":
            fig = px.histogram(df, x=field, nbins=45, title=f"Distribution of {label.lower()}", labels={field: label}, color_discrete_sequence=["#2563EB"])
            stats.update({"Mean": _format(series.mean(), field), "Median": _format(series.median(), field), "25th–75th percentile": f"{_format(series.quantile(.25), field)} – {_format(series.quantile(.75), field)}"})
            answer = f"Across {len(series):,} filtered properties, the median {label.lower()} is {_format(series.median(), field)} and the mean is {_format(series.mean(), field)}."
            interpretation = "The histogram shows the observed values in the current selection; extreme observations can pull the mean away from the median."
        elif kind in {"scalar", "summary"}:
            stats.update({"Mean": _format(series.mean(), field), "Median": _format(series.median(), field), "Minimum": _format(series.min(), field), "Maximum": _format(series.max(), field)})
            if kind == "scalar":
                value = series.agg(spec["agg"])
                name = "Average" if spec["agg"] == "mean" else spec["agg"].title()
                answer = f"The {name.lower()} {label.lower()} in the filtered dataset is {_format(value, field)}."
            else:
                answer = f"For the filtered properties, average size is {_format(series.mean(), field)} and median size is {_format(series.median(), field)}."
            fig = px.histogram(df, x=field, nbins=35, title=f"{label} in filtered properties", labels={field: label}, color_discrete_sequence=["#2563EB"])
            interpretation = "These are descriptive statistics of the current listings, not transaction outcomes."
        elif kind == "common_bin":
            bins = int(spec.get("bins", 10))
            cut = pd.cut(series, bins=bins, include_lowest=True, duplicates="drop")
            chart_data = cut.value_counts(sort=False).rename_axis("Range").reset_index(name="Properties")
            mode_row = chart_data.loc[chart_data.Properties.idxmax()]
            stats["Most common range"] = str(mode_row.Range)
            stats["Properties in range"] = f"{int(mode_row.Properties):,}"
            chart_data["Range"] = chart_data["Range"].astype(str)
            answer = f"The most populated {label.lower()} band is {mode_row.Range}, containing {int(mode_row.Properties):,} filtered properties."
            fig = px.bar(chart_data, x="Range", y="Properties", title=f"Properties by {label.lower()} band", color_discrete_sequence=["#2563EB"])
            interpretation = "Equal-width bins summarize the observed range; bin boundaries are descriptive and not market-defined price or size tiers."
        else:
            current = _metric(df, field, "median")
            all_value = _metric(full_df, field, "median")
            chart_data = pd.DataFrame({"Dataset": ["Current filtered selection", "Full dataset"], "Median": [current, all_value]})
            stats.update({"Filtered median": _format(current, field), "Full dataset median": _format(all_value, field), "Filtered records": f"{len(df):,}", "Full records": f"{len(full_df):,}"})
            answer = f"The filtered median {label.lower()} is {_format(current, field)}, compared with {_format(all_value, field)} across all {len(full_df):,} rows."
            fig = px.bar(chart_data, x="Dataset", y="Median", title=f"Filtered versus full-dataset median {label.lower()}", labels={"Median": label}, color="Dataset", color_discrete_sequence=["#2563EB", "#0F766E"])
            interpretation = "A difference reflects the current filter selection and its composition; it does not imply a change over time."

    elif kind in {"group", "share"}:
        group = spec["group"]
        if kind == "share":
            chart_data = df[group].value_counts().rename_axis(group).reset_index(name="Properties")
            chart_data["Share"] = 100 * chart_data.Properties / len(df)
            chart_data = chart_data.sort_values("Share", ascending=False)
            stats["Largest group"] = str(chart_data.iloc[0][group])
            stats["Largest share"] = f"{chart_data.iloc[0].Share:.1f}%"
            answer = f"{chart_data.iloc[0][group]} represents {chart_data.iloc[0].Share:.1f}% of the filtered listings ({int(chart_data.iloc[0].Properties):,} properties)."
            fig = px.bar(chart_data, x=group, y="Share", title=f"Listing share by {group.replace('_', ' ').lower()} (%)", labels={"Share": "Share of filtered listings (%)"}, color_discrete_sequence=["#2563EB"])
        else:
            agg = spec.get("agg", "median")
            if agg == "count":
                chart_data = df.groupby(group, dropna=False).size().rename("Value").reset_index().rename(columns={"Value": "Properties"})
                value_col, label = "Properties", "Properties"
            else:
                chart_data = df.groupby(group, dropna=False)[field].agg(Value=agg, Properties="size").reset_index()
                value_col, label = "Value", FIELD_LABELS.get(field, field)
            chart_data = chart_data[chart_data.Properties >= int(spec.get("min_count", 1))]
            if chart_data.empty:
                empty["answer"] = "No category meets the minimum sample size for this comparison."
                empty["caveat"] = f"Groups with fewer than {int(spec.get('min_count', 1))} listings are omitted."
                return empty
            chart_data = chart_data.sort_values(value_col, ascending=False)
            rank = spec.get("rank")
            if rank == "lowest":
                chart_data = chart_data.nsmallest(int(spec.get("top", 15)), value_col)
            elif rank == "spread":
                chart_data = pd.concat([chart_data.head(1), chart_data.tail(1)]).drop_duplicates(group)
            else:
                chart_data = chart_data.head(int(spec.get("top", 20)))
            plot_data = chart_data.sort_values(value_col)
            if rank == "lowest":
                ordered = chart_data.sort_values(value_col)
            else:
                ordered = plot_data
            fig = px.bar(ordered, x=value_col, y=group, orientation="h", title=f"{agg.title()} {label.lower()} by {group.replace('_', ' ').lower()}", labels={value_col: label}, hover_data=["Properties"], color_discrete_sequence=["#2563EB"])
            first = chart_data.iloc[0]
            if rank == "spread" and len(chart_data) > 1:
                low = chart_data.iloc[-1]
                difference = float(first[value_col] - low[value_col])
                stats.update({"Highest category": str(first[group]), "Highest average": _format(first[value_col], field), "Lowest category": str(low[group]), "Lowest average": _format(low[value_col], field), "Difference": _format(difference, field)})
                answer = f"Among cities with at least {int(spec.get('min_count', 1))} listings, {first[group]} has the highest average ({_format(first[value_col], field)}) and {low[group]} the lowest ({_format(low[value_col], field)}); the gap is {_format(difference, field)}."
            elif rank == "spread":
                only = chart_data.iloc[0]
                stats.update({"Only eligible city": str(only[group]), "Average price": _format(only[value_col], field), "Listings": f"{int(only.Properties):,}"})
                answer = f"Only {only[group]} has enough listings for this comparison under the current filters, so a city-to-city price gap cannot be calculated."
            else:
                extreme = chart_data.iloc[0] if rank == "lowest" else first
                stat_name = "Count" if agg == "count" else agg.title()
                stats.update({f"Highest {stat_name.lower()} category" if rank != "lowest" else f"Lowest {stat_name.lower()} category": str(extreme[group]), stat_name: _format(extreme[value_col], field) if agg != "count" else f"{int(extreme[value_col]):,}", "Listings represented": f"{int(extreme.Properties):,}"})
                direction = "lowest" if rank == "lowest" else "highest"
                answer = f"{extreme[group]} has the {direction} {stat_name.lower()} among the displayed groups ({_format(extreme[value_col], field) if agg != 'count' else f'{int(extreme[value_col]):,} properties'}; {int(extreme.Properties):,} listings)."
        interpretation = "Group averages and medians are unadjusted descriptive comparisons. Differences may reflect other property or location characteristics."
        caveat = f"Categories with fewer than {int(spec.get('min_count', 1))} listings are excluded." if spec.get("min_count", 1) > 1 else ""

    elif kind == "group_box":
        field, group = spec["field"], spec["group"]
        fig = px.box(df, x=group, y=field, color=group, title=f"{FIELD_LABELS.get(field, field)} by {group.replace('_', ' ').lower()}", labels={field: FIELD_LABELS.get(field, field)})
        chart_data = df[[group, field]].copy()
        summary = df.groupby(group)[field].agg(Median="median", Mean="mean", Listings="size").reset_index()
        stats.update({"Highest group median": str(summary.loc[summary.Median.idxmax(), group]), "Highest group median value": _format(summary.Median.max(), field)})
        answer = f"{summary.loc[summary.Median.idxmax(), group]} has the highest group median ({_format(summary.Median.max(), field)})."
        interpretation = "Box plots show medians, spread and potential outliers; category differences are not adjusted for other property attributes."
        chart_data = summary

    elif kind in {"scatter", "pair_correlation"}:
        x, y = spec["x"], spec["y"]
        r = df[[x, y]].corr().iloc[0, 1]
        sample = df.sample(min(12000, len(df)), random_state=42) if len(df) > 12000 else df
        chart_data = sample[[x, y]].copy()
        fig = px.scatter(sample, x=x, y=y, opacity=.55, labels={x: FIELD_LABELS.get(x, x.replace("_", " ")), y: FIELD_LABELS.get(y, y.replace("_", " "))}, title=f"{FIELD_LABELS.get(x, x)} versus {FIELD_LABELS.get(y, y)}", color_discrete_sequence=["#2563EB"])
        stats["Pearson correlation"] = f"{r:.3f}" if pd.notna(r) else "Unavailable (insufficient variation or records)"
        answer = f"The Pearson correlation between {FIELD_LABELS.get(x, x.lower())} and {FIELD_LABELS.get(y, y.lower())} is {r:.3f} for the filtered records." if pd.notna(r) else "There is not enough variation in the filtered records to calculate a Pearson correlation."
        interpretation = "The scatter plot shows association. Correlation does not establish causality, and extreme values can affect Pearson's coefficient."
        caveat = f"Chart displays a deterministic sample of {len(sample):,} from {len(df):,} rows when the filtered dataset is large." if len(df) > 12000 else ""

    elif kind == "binned_group":
        field, by = spec["field"], spec["by"]
        band = f"{by.replace('_', ' ')} band"
        temp = df[[by, field]].copy()
        temp[band] = pd.cut(temp[by], bins=int(spec.get("bins", 10)), include_lowest=True, duplicates="drop")
        chart_data = temp.groupby(band, observed=False)[field].agg(Median="median", Mean="mean", Listings="size").reset_index().dropna(subset=["Median"])
        chart_data[band] = chart_data[band].astype(str)
        agg = "Median" if field == "Price_in_Lakhs" else "Mean"
        value = chart_data.loc[chart_data[agg].idxmax()]
        fig = px.bar(chart_data, x=band, y=agg, title=f"{agg} {FIELD_LABELS.get(field, field)} across {band}s", labels={agg: FIELD_LABELS.get(field, field)}, color_discrete_sequence=["#0F766E"])
        answer = f"The highest {agg.lower()} {FIELD_LABELS.get(field, field).lower()} occurs in band {value[band]} ({_format(value[agg], field)})."
        stats.update({"Highest band": value[band], f"Highest {agg.lower()}": _format(value[agg], field), "Properties in band": f"{int(value.Listings):,}"})
        interpretation = "Numeric ranges are split into equal-width bins for comparison; this is descriptive rather than a market-defined segmentation."

    elif kind in {"outlier", "outlier_compare", "outlier_box_both"}:
        fields = [spec["field"]] if kind == "outlier" else ["Price_per_SqFt_Clean", "Size_in_SqFt"]
        records = []
        for out_field in fields:
            count, low, high = _outlier_stats(df[out_field].dropna())
            records.append({"Measure": FIELD_LABELS.get(out_field, out_field), "Potential outliers": count, "Share (%)": 100 * count / len(df), "Lower fence": low, "Upper fence": high})
        chart_data = pd.DataFrame(records)
        if kind == "outlier_box_both":
            figures = [
                px.box(df, y=out_field, title=f"Potential outliers: {FIELD_LABELS.get(out_field, out_field)}", labels={out_field: FIELD_LABELS.get(out_field, out_field)}, color_discrete_sequence=["#2563EB"])
                for out_field in fields
            ]
            answer = "The two box plots show each measure on its own scale; the table reports 1.5×IQR fences and the number of points outside them."
        else:
            fig = px.bar(chart_data, x="Measure", y="Potential outliers", title="Potential outliers by 1.5×IQR rule", color_discrete_sequence=["#2563EB"])
            answer = "; ".join(f"{row['Measure']}: {int(row['Potential outliers']):,} potential outliers ({row['Share (%)']:.2f}%)" for _, row in chart_data.iterrows()) + "."
            if kind == "outlier":
                stats.update({"Potential outliers": f"{int(chart_data.iloc[0]['Potential outliers']):,}", "Share": f"{chart_data.iloc[0]['Share (%)']:.2f}%", "Lower fence": _format(chart_data.iloc[0]["Lower fence"], field), "Upper fence": _format(chart_data.iloc[0]["Upper fence"], field)})
        interpretation = "The 1.5×IQR rule flags statistical extremes, not confirmed data errors. The source rows remain included."

    elif kind == "correlation_heatmap":
        corr = correlation_data(df)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation matrix for numeric variables")
        chart_data = corr.rename_axis("Variable").reset_index()
        answer = f"The matrix compares Pearson correlations across {corr.shape[0]} numeric variables in the current selection."
        stats["Numeric variables"] = f"{corr.shape[0]}"
        interpretation = "Correlations measure linear association and do not show causality. Derived or closely related variables may correlate by construction."
    elif kind == "strongest_correlation":
        corr = correlation_data(df)
        abs_corr = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric variable correlation matrix")
        chart_data = corr.rename_axis("Variable").reset_index()
        pairs = abs_corr.stack()
        if pairs.empty or pairs.dropna().empty:
            answer = "There are not enough records with variation across numeric fields to identify a strongest correlation."
            stats["Strongest pair"] = "Unavailable"
        else:
            pair = pairs.idxmax()
            value = corr.loc[pair[0], pair[1]]
            answer = f"The strongest absolute linear association is between {pair[0].replace('_', ' ')} and {pair[1].replace('_', ' ')} (Pearson r = {value:.3f})."
            stats["Strongest pair"] = f"{pair[0]} × {pair[1]}"
            stats["Pearson correlation"] = f"{value:.3f}"
        interpretation = "This is a descriptive correlation, not a causal or investment relationship."

    elif kind in {"facility_association", "facility_group", "facility_counts"}:
        facility = spec["field"]
        chart_data = df.groupby(facility).agg(Median_rate=("Price_per_SqFt_Clean", "median"), Mean_rate=("Price_per_SqFt_Clean", "mean"), Listings=("ID", "size")).reset_index()
        rho = df[[facility, "Price_per_SqFt_Clean"]].corr(method="spearman").iloc[0, 1]
        if kind == "facility_counts":
            fig = px.bar(chart_data, x=facility, y="Listings", title=f"Listings by {facility.replace('_', ' ').lower()}", labels={"Listings": "Properties"}, color_discrete_sequence=["#2563EB"])
            most = chart_data.loc[chart_data.Listings.idxmax()]
            answer = f"The most represented count is {int(most[facility])} {facility.replace('_', ' ').lower()} ({int(most.Listings):,} listings)."
        else:
            fig = px.line(chart_data, x=facility, y="Median_rate", markers=True, title=f"Median corrected rate by {facility.replace('_', ' ').lower()}", labels={"Median_rate": "Median corrected price (₹/sq ft)"}, color_discrete_sequence=["#0F766E"])
            peak = chart_data.loc[chart_data.Median_rate.idxmax()]
            answer = f"Median corrected rate ranges from ₹{chart_data.Median_rate.min():,.0f} to ₹{chart_data.Median_rate.max():,.0f} per sq ft across observed {facility.replace('_', ' ').lower()} counts; the maximum is at {int(peak[facility])}."
            stats["Spearman association"] = f"{rho:.3f}"
            stats["Highest median-rate count"] = f"{int(peak[facility])}"
        interpretation = "This comparison shows association only; it does not establish that nearby facilities cause a price difference."

    elif kind == "city_bhk_heatmap":
        matrix = pd.crosstab(df.City, df.BHK)
        fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", title="Listing counts by city and BHK")
        chart_data = matrix.rename_axis("City").reset_index()
        mode = df.BHK.mode().iloc[0]
        answer = f"The heatmap shows {len(df.City.unique())} cities by BHK configuration. The most common configuration overall is {mode} BHK."
        stats["Cities"] = f"{df.City.nunique():,}"
        stats["Most common BHK"] = f"{mode} BHK"
        interpretation = "Counts reflect the filtered listings. Cities with few records may show less stable distributions."

    elif kind == "city_bhk_mode":
        counts = df.groupby(["City", "BHK"]).size().rename("Properties").reset_index()
        chart_data = counts.loc[counts.groupby("City").Properties.idxmax()].sort_values("Properties", ascending=False).head(25)
        fig = px.bar(chart_data.sort_values("Properties"), x="Properties", y="City", orientation="h", color="BHK", title="Most common BHK configuration by city", labels={"BHK": "Most common BHK"})
        first = chart_data.iloc[0]
        answer = f"The most listings in a city's leading BHK group occur in {first.City}: {int(first.Properties):,} properties with {first.BHK} BHK."
        stats["Cities shown"] = f"{len(chart_data):,}"
        interpretation = "This reports the most frequent BHK in each city, not the full distribution or the share of each configuration."

    elif kind in {"top_localities", "top_locality_medians", "top_locality_counts", "locality_year_trend"}:
        top = df.groupby("Locality").Price_in_Lakhs.mean().nlargest(5).index
        local = df[df.Locality.isin(top)]
        if kind == "top_localities":
            chart_data = local.groupby("Locality").Price_in_Lakhs.agg(Mean="mean", Median="median", Listings="size").sort_values("Mean", ascending=False).reset_index()
            fig = px.bar(chart_data.sort_values("Mean"), x="Mean", y="Locality", orientation="h", hover_data=["Median", "Listings"], title="Five localities with highest mean listing price", labels={"Mean": "Mean price (₹ lakh)"}, color_discrete_sequence=["#2563EB"])
            first = chart_data.iloc[0]
            answer = f"{first.Locality} ranks highest by mean listing price at ₹{first.Mean:,.1f} lakh ({int(first.Listings):,} listings)."
        elif kind == "top_locality_medians":
            chart_data = local.groupby("Locality").Price_in_Lakhs.agg(Median="median", Mean="mean", Listings="size").sort_values("Median", ascending=False).reset_index()
            fig = px.bar(chart_data.sort_values("Median"), x="Median", y="Locality", orientation="h", hover_data=["Mean", "Listings"], title="Median price in the five highest-mean localities", labels={"Median": "Median price (₹ lakh)"}, color_discrete_sequence=["#0F766E"])
            answer = f"Within these five localities, {chart_data.iloc[0].Locality} has the highest median listing price (₹{chart_data.iloc[0].Median:,.1f} lakh)."
        elif kind == "top_locality_counts":
            chart_data = local.groupby("Locality").size().rename("Listings").sort_values(ascending=False).reset_index()
            fig = px.bar(chart_data.sort_values("Listings"), x="Listings", y="Locality", orientation="h", title="Listings in the five highest-mean localities", color_discrete_sequence=["#2563EB"])
            answer = f"{chart_data.iloc[0].Locality} has the largest representation among the five selected localities ({int(chart_data.iloc[0].Listings):,} listings)."
        else:
            chart_data = local.groupby(["Year_Built", "Locality"]).Price_in_Lakhs.median().rename("Median_price").reset_index()
            fig = px.line(chart_data, x="Year_Built", y="Median_price", color="Locality", markers=True, title="Median listing price by construction year", labels={"Year_Built": "Year built", "Median_price": "Median price (₹ lakh)"})
            answer = "The chart compares current listing prices across construction-year groups in the five localities with the highest mean listing prices."
        stats["Localities selected"] = ", ".join(map(str, top))
        interpretation = "Locality labels such as Locality_84 are dataset identifiers rather than descriptive geographic names. Construction year is not transaction time; these are cross-sectional comparisons, not historical price trends."
        caveat = "Small construction-year groups can be noisy; hover to inspect group values." if kind == "locality_year_trend" else "Localities were selected by mean price within the current filtered dataset."

    elif kind in {"parking_security_heatmap", "investment_limit"}:
        if kind == "parking_security_heatmap":
            matrix = df.pivot_table(index="Parking_Space", columns="Security", values="Price_in_Lakhs", aggfunc="median")
            fig = px.imshow(matrix, text_auto=".1f", color_continuous_scale="Blues", title="Median price by parking and security", labels={"color": "Median ₹ lakh"})
            chart_data = matrix.rename_axis("Parking").reset_index()
            answer = "The heatmap reports median listing prices for each parking and security combination in the filtered data."
            interpretation = "These are unadjusted group medians; size, location and property type may differ across groups."
        else:
            chart_data = df.groupby("Public_Transport_Accessibility").agg(Median_price=("Price_in_Lakhs", "median"), Median_rate=("Price_per_SqFt_Clean", "median"), Listings=("ID", "size")).reset_index()
            fig = px.bar(chart_data, x="Public_Transport_Accessibility", y="Median_price", color="Public_Transport_Accessibility", hover_data=["Median_rate", "Listings"], title="Listing price by transport-access label", labels={"Median_price": "Median price (₹ lakh)"})
            answer = "The dataset supports descriptive comparisons of listing price and corrected price per sq ft across transport-access labels. It does not contain rent, transaction returns, appreciation history, or a verified investment outcome."
            interpretation = "Transport access comparisons do not establish causation and cannot support a definitive investment recommendation."
            caveat = "Investment returns, rental yield, liquidity, and legal or infrastructure quality cannot be assessed from the available columns."

    elif kind in {"amenities", "amenity_counts", "amenities_box"}:
        grouped = df.groupby("Amenities_Clean").agg(Median_rate=("Price_per_SqFt_Clean", "median"), Mean_rate=("Price_per_SqFt_Clean", "mean"), Listings=("ID", "size")).reset_index()
        if kind == "amenity_counts":
            chart_data = grouped.nlargest(15, "Listings").sort_values("Listings")
            fig = px.bar(chart_data, x="Listings", y="Amenities_Clean", orientation="h", title="Most common standardized amenity combinations", color_discrete_sequence=["#2563EB"])
            answer = f"The most frequent amenity combination is {grouped.loc[grouped.Listings.idxmax(), 'Amenities_Clean']} ({int(grouped.Listings.max()):,} listings)."
        else:
            measure = "Median_rate" if spec.get("agg") == "median" else "Mean_rate"
            chart_data = grouped[grouped.Listings >= 30].sort_values(measure, ascending=False).head(15)
            if chart_data.empty:
                empty["answer"] = "No amenity combination has at least 30 listings under the current filters."
                empty["caveat"] = "Broaden the filters to include more observations for amenity comparisons."
                return empty
            if kind == "amenities_box":
                chosen = chart_data.head(8).Amenities_Clean.tolist()
                source = df[df.Amenities_Clean.isin(chosen)]
                fig = px.box(source, x="Amenities_Clean", y="Price_per_SqFt_Clean", title="Corrected rate for common amenity combinations", labels={"Price_per_SqFt_Clean": "Corrected price (₹/sq ft)"})
                chart_data = source.groupby("Amenities_Clean").Price_per_SqFt_Clean.agg(Median="median", Listings="size").reset_index()
                answer = "The box plots compare corrected-rate distributions for common amenity combinations with at least 30 observations."
            else:
                chart_data = chart_data.sort_values(measure)
                fig = px.bar(chart_data, x=measure, y="Amenities_Clean", orientation="h", title="Corrected price per sq ft by amenity combination (30+ listings)", labels={"Median_rate": "Median corrected rate (₹/sq ft)", "Mean_rate": "Mean corrected rate (₹/sq ft)"}, hover_data=["Listings"], color_discrete_sequence=["#0F766E"])
                peak = chart_data.iloc[-1]
                answer = f"{peak.Amenities_Clean} has the highest displayed {spec.get('agg')} corrected rate (₹{peak[measure]:,.0f}/sq ft; {int(peak.Listings):,} listings)."
        interpretation = "Amenity combinations are descriptive groups. Their rates do not estimate the independent value of an amenity."
        caveat = "Only combinations with at least 30 filtered listings are included in rate comparisons."

    if not answer:
        answer = "The selected comparison is summarized in the chart and statistics below."
    if not isinstance(chart_data, pd.DataFrame) or chart_data.empty:
        chart_data = pd.DataFrame([{"Statistic": key, "Value": value} for key, value in stats.items()])
    return {
        "answer": answer,
        "stats": stats,
        "figure": apply_chart_theme(fig) if fig is not None else None,
        "figures": [apply_chart_theme(figure) for figure in figures],
        "table": chart_data,
        "interpretation": interpretation,
        "caveat": caveat,
    }
