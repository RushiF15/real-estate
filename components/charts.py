import plotly.express as px

NAVY = "#14213D"
BLUE = "#2563EB"
TEAL = "#0F766E"
SLATE = "#64748B"
GRID = "#E9EEF5"
BORDER = "#CBD5E1"
COLORWAY = [BLUE, TEAL, NAVY, "#7C3AED", "#B45309", "#15803D", "#B91C1C"]


def apply_chart_theme(fig):
    """Apply the shared high-contrast light theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        colorway=COLORWAY,
        font={"family": "DM Sans, sans-serif", "size": 12, "color": "#334155"},
        title={"font": {"family": "Manrope, sans-serif", "size": 16, "color": NAVY}, "x": 0.02, "xanchor": "left"},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin={"l": 18, "r": 18, "t": 55, "b": 22},
        legend={"font": {"color": "#334155", "size": 11}, "title": {"font": {"color": NAVY}}},
        coloraxis_colorbar={"tickfont": {"color": "#475569"}, "title": {"font": {"color": SLATE}}},
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": BORDER,
            "font": {"family": "DM Sans, sans-serif", "size": 12, "color": "#172033"},
            "align": "left",
        },
    )
    axis_style = {
        "title_font": {"family": "DM Sans, sans-serif", "size": 12, "color": SLATE},
        "tickfont": {"family": "DM Sans, sans-serif", "size": 11, "color": "#475569"},
        "gridcolor": GRID,
        "linecolor": BORDER,
        "zerolinecolor": GRID,
        "showline": True,
        "automargin": True,
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


# Kept as a compatibility alias for any external imports from earlier versions.
base = apply_chart_theme


def price_histogram(df):
    fig = px.histogram(
        df,
        x="Price_in_Lakhs",
        nbins=40,
        title="Property price distribution (₹ lakh)",
        labels={"Price_in_Lakhs": "Price (₹ lakh)", "count": "Properties"},
        color_discrete_sequence=[BLUE],
    )
    return apply_chart_theme(fig)


def count_bar(df, col, title, top=20):
    data = df[col].value_counts().head(top).rename_axis(col).reset_index(name="Properties")
    fig = px.bar(data, x="Properties", y=col, orientation="h", title=title, color_discrete_sequence=[BLUE])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return apply_chart_theme(fig)


def categorical_bar(df, group, value, title, aggregation="median", top=20):
    data = df.groupby(group)[value].agg(aggregation).sort_values(ascending=False).head(top).rename("value").reset_index()
    fig = px.bar(
        data,
        x=group,
        y="value",
        title=title,
        labels={"value": f"{aggregation.title()} ₹ lakh"},
        color_discrete_sequence=[TEAL],
    )
    return apply_chart_theme(fig)
