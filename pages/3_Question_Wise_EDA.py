import streamlit as st
import pandas as pd
from html import escape

from components.styling import apply_style
from src.data_loader import load_data
from src.sidebar import render_sidebar
from src.eda_explorer import QUESTION_CATEGORIES, QUESTIONS, build_answer


apply_style()
st.title("Question-wise EDA")
st.caption("Explore the housing data through 20 guided questions. Select a topic, then choose a subquestion to see a filtered, computed answer.")

df = load_data()
if df is None:
    st.error("Dataset missing. Configure HOUSING_DATA_PATH or place the cleaned CSV in the project folder.")
    st.stop()

filtered = render_sidebar(df)
st.caption(f"Current EDA selection: **{len(filtered):,}** matching properties out of {len(df):,}. All answers below use these active filters unless a full-dataset comparison is explicitly shown.")

if "eda_main_question" not in st.session_state or st.session_state.eda_main_question not in QUESTIONS:
    st.session_state.eda_main_question = "q1"
if "eda_subquestion" not in st.session_state:
    st.session_state.eda_subquestion = 0

nav_col, answer_col = st.columns([.9, 2.1], gap="large")
with nav_col:
    st.markdown("#### Question explorer")
    if st.button("Reset question selection", key="eda_reset_questions", use_container_width=True, help="Return to the first EDA question. Global filters are unchanged."):
        st.session_state.eda_main_question = "q1"
        st.session_state.eda_subquestion = 0
        st.session_state.pop("eda_sub_radio_q1", None)
        st.rerun()

    for category in QUESTION_CATEGORIES:
        is_open = any(q["id"] == st.session_state.eda_main_question for q in category["questions"])
        with st.expander(category["category"], expanded=is_open):
            for question in category["questions"]:
                is_selected = question["id"] == st.session_state.eda_main_question
                button_label = f"{'● ' if is_selected else ''}{question['id'].upper()}. {question['title']}"
                if st.button(button_label, key=f"eda_main_{question['id']}", use_container_width=True, type="primary" if is_selected else "secondary"):
                    st.session_state.eda_main_question = question["id"]
                    st.session_state.eda_subquestion = 0
                    st.session_state.pop(f"eda_sub_radio_{question['id']}", None)
                    st.rerun()

    selected_question = QUESTIONS[st.session_state.eda_main_question]
    subquestions = selected_question["subquestions"]
    if st.session_state.eda_subquestion >= len(subquestions):
        st.session_state.eda_subquestion = 0
    st.markdown("#### Subquestions")
    sub_labels = [sub["label"] for sub in subquestions]
    selected_label = st.radio(
        "Choose what to explore",
        sub_labels,
        index=st.session_state.eda_subquestion,
        key=f"eda_sub_radio_{selected_question['id']}",
        label_visibility="collapsed",
    )
    st.session_state.eda_subquestion = sub_labels.index(selected_label)

with answer_col:
    if filtered.empty:
        st.info("No properties match the current global filters. Adjust the filters in the sidebar to explore answers.")
        st.stop()

    selected_question = QUESTIONS[st.session_state.eda_main_question]
    selected_sub = selected_question["subquestions"][st.session_state.eda_subquestion]
    st.markdown(f"<div class='eyebrow'>{selected_question['id'].upper()} · {selected_question['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"### {selected_sub['label']}")

    with st.spinner("Computing the answer from the filtered properties…"):
        result = build_answer(filtered, df, selected_sub)

    st.markdown(f"<div class='insight-card'><strong>Answer</strong><br>{escape(result['answer'])}</div>", unsafe_allow_html=True)
    if result["stats"]:
        metric_cols = st.columns(min(4, len(result["stats"])))
        for i, (label, value) in enumerate(result["stats"].items()):
            metric_cols[i % len(metric_cols)].metric(label, value)

    if result.get("figure") is not None:
        st.plotly_chart(result["figure"], use_container_width=True, key=f"eda_chart_{selected_question['id']}_{st.session_state.eda_subquestion}")
    for i, figure in enumerate(result.get("figures", [])):
        st.plotly_chart(figure, use_container_width=True, key=f"eda_chart_{selected_question['id']}_{st.session_state.eda_subquestion}_{i}")

    if result["interpretation"]:
        st.markdown(f"**Interpretation:** {result['interpretation']}")
    if result["caveat"]:
        st.caption(f"Limitation: {result['caveat']}")

    chart_data = result["table"]
    if isinstance(chart_data, pd.DataFrame) and not chart_data.empty:
        with st.expander("View supporting data", expanded=False):
            st.dataframe(chart_data.head(100), use_container_width=True, hide_index=True)
            st.caption(f"Showing up to 100 of {len(chart_data):,} rows in the table. The CSV download contains the complete supporting table.")
        st.download_button(
            "Download supporting data (CSV)",
            chart_data.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_question['id']}_{st.session_state.eda_subquestion + 1}_eda.csv",
            mime="text/csv",
            key=f"eda_download_{selected_question['id']}_{st.session_state.eda_subquestion}",
        )
