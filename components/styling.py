import streamlit as st


def apply_style():
    st.markdown(
        """<style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root {
          --navy: #14213D; --blue: #2563EB; --body: #334155; --muted: #64748B;
          --page: #F5F7FB; --panel: #FFFFFF; --input-text: #172033; --border: #E2E8F0;
        }
        .stApp { background: var(--page); color: var(--body); font-family: 'DM Sans', sans-serif; }
        [data-testid="stHeader"] { background: rgba(245,247,251,.96); }
        section.main > div { padding-top: 2rem; max-width: 1500px; }
        [data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: var(--body); }
        [data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] { color: var(--body) !important; border-radius: 8px; }
        [data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNavLink"]:hover { color: var(--navy) !important; background: #EFF4FF; }
        [data-testid="stSidebarNav"] a[aria-current="page"], [data-testid="stSidebarNavLink"][aria-current="page"] { color: var(--blue) !important; background: #EFF4FF; font-weight: 700; }
        [data-testid="stSidebarNav"] a:focus-visible,
        [data-testid="stSidebar"] button:focus-visible,
        .stApp button:focus-visible,
        [data-baseweb="select"]:focus-within,
        input:focus-visible, textarea:focus-visible { outline: 3px solid rgba(37,99,235,.38) !important; outline-offset: 2px; }

        h1, h2, h3, h4, h5, h6 { color: var(--navy) !important; font-family: 'Manrope', sans-serif; }
        h1 { font-size: 2rem !important; letter-spacing: -.035em; line-height: 1.2 !important; }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li { color: var(--body); }
        [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
        [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
        [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
        label, [data-testid="stCheckbox"] label, [data-testid="stRadio"] label { color: var(--body); }
        .eyebrow { color: var(--blue); letter-spacing: 1.4px; font-size: .7rem; font-weight: 700; margin: 0 0 7px; }
        .section-heading { font: 700 1.12rem 'Manrope', sans-serif; color: var(--navy); margin: 1.5rem 0 .7rem; }

        [data-testid="stMetric"] { background: var(--panel); padding: 16px 18px; border: 1px solid var(--border); border-radius: 13px; box-shadow: 0 2px 9px rgba(20,33,61,.035); }
        [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .78rem !important; line-height: 1.3; white-space: normal !important; overflow-wrap: anywhere; }
        [data-testid="stMetricLabel"] p { color: var(--muted) !important; white-space: normal !important; }
        [data-testid="stMetricValue"] { color: var(--navy) !important; font: 700 1.42rem 'Manrope', sans-serif; }
        [data-testid="stMetricDelta"] { color: var(--body); }
        .insight-card { background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--blue); border-radius: 9px; padding: 12px 15px; margin: .45rem 0; color: var(--body); font-size: .91rem; }
        .insight-card strong { color: var(--navy); }

        [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea { background: var(--panel) !important; color: var(--input-text) !important; border-color: var(--border) !important; }
        [data-baseweb="select"] *, [data-baseweb="input"] *, [data-baseweb="textarea"] * { color: var(--input-text); }
        [role="listbox"], [role="listbox"] [role="option"], [data-baseweb="menu"],
        [data-baseweb="menu"] [role="option"], [data-baseweb="popover"] ul,
        [data-baseweb="popover"] li { background: var(--panel) !important; color: var(--input-text) !important; }
        [role="option"]:hover, [role="option"][aria-selected="true"] { background: #EFF4FF !important; color: var(--navy) !important; }
        [data-testid="stSlider"] [role="slider"] { border-color: var(--blue); }

        .stApp button[kind="primary"], .stApp button[data-testid="stBaseButton-primary"] { background: var(--blue); border: 1px solid var(--blue); color: #FFFFFF; }
        .stApp button[kind="primary"] p, .stApp button[data-testid="stBaseButton-primary"] p { color: #FFFFFF !important; }
        .stApp button[kind="secondary"], [data-testid="stSidebar"] button { color: var(--navy); border-color: var(--border); background: #FFFFFF; }
        .stApp button[kind="secondary"]:hover, [data-testid="stSidebar"] button:hover { color: var(--navy); border-color: var(--blue); background: #EFF4FF; }

        [data-testid="stAlert"] { color: var(--body); border-radius: 10px; }
        [data-testid="stAlert"] p, [data-testid="stAlert"] li { color: var(--body) !important; }
        [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--border); border-radius: 10px; background: #FFFFFF; }
        [data-testid="stPlotlyChart"] { border: 1px solid var(--border); border-radius: 12px; background: #FFFFFF; padding: 5px; }
        .brand-mark { width: 38px; height: 38px; border-radius: 11px; background: var(--navy); color: #FFFFFF; display: inline-flex; align-items: center; justify-content: center; font-weight: 800; font-family: 'Manrope', sans-serif; margin-right: 9px; vertical-align: middle; }
        .brand-title { display: inline-block; font: 700 16px 'Manrope', sans-serif; color: var(--navy); vertical-align: middle; }
        .brand-sub { margin: 9px 0 6px 48px; font-size: 9px; letter-spacing: 1.25px; color: var(--muted); }
        .sidebar-foot { font-size: .72rem; color: var(--muted); line-height: 1.6; margin-top: 2rem; }
        @media (max-width: 760px) { h1 { font-size: 1.6rem !important; } [data-testid="stMetric"] { padding: 12px; } .section-heading { margin-top: 1rem; } }
        </style>""",
        unsafe_allow_html=True,
    )
