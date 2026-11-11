from pathlib import Path
import math

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# ChurnIQ | Telco Customer Churn Analytics
# A polished Streamlit portfolio application.
# Keep these files beside app.py:
#   - churn_pipeline.joblib
#   - WA_Fn-UseC_-Telco-Customer-Churn.csv
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_pipeline.joblib"
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme -----------------------------
BG = "#080F1C"
SIDEBAR = "#0B1423"
PANEL = "#101C2E"
PANEL_ALT = "#14243A"
BORDER = "#21334B"
TEXT = "#EAF1FA"
MUTED = "#91A4BC"
CYAN = "#42C8F5"
BLUE = "#648BFF"
RED = "#FF718B"
GREEN = "#38D9A9"
AMBER = "#F6BD60"
PURPLE = "#A78BFA"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {{
  --bg: {BG}; --panel: {PANEL}; --panel-alt: {PANEL_ALT};
  --border: {BORDER}; --text: {TEXT}; --muted: {MUTED};
  --cyan: {CYAN}; --red: {RED}; --green: {GREEN};
}}
html, body, [class*="css"], .stApp {{
  font-family: 'DM Sans', sans-serif;
}}
.stApp {{
  background:
    radial-gradient(900px 500px at 90% -10%, rgba(66,200,245,.09), transparent 62%),
    radial-gradient(700px 420px at -12% 100%, rgba(167,139,250,.06), transparent 60%),
    {BG};
  color: {TEXT};
}}
#MainMenu, footer {{visibility:hidden;}}
header[data-testid="stHeader"] {{background:transparent;}}
.block-container {{max-width:1500px; padding-top:1.8rem; padding-bottom:3.5rem;}}
h1,h2,h3,h4 {{color:{TEXT}!important; letter-spacing:-.035em;}}
p, label, .stCaption {{color:{MUTED};}}
hr {{border-color:{BORDER};}}
[data-testid="stSidebar"] {{
  background:{SIDEBAR}; border-right:1px solid {BORDER};
}}
[data-testid="stSidebar"] > div:first-child {{padding-top:1.4rem;}}
.side-brand {{display:flex; align-items:center; gap:12px; padding:4px 0 24px;}}
.brand-mark {{
  width:44px;height:44px;border-radius:14px;display:flex;align-items:center;
  justify-content:center;background:linear-gradient(135deg,{CYAN},{BLUE});
  color:#06111E;font-size:22px;font-weight:800;
  box-shadow:0 8px 28px rgba(66,200,245,.18);
}}
.brand-name {{font-size:21px;font-weight:800;color:{TEXT};line-height:1.05;}}
.brand-caption {{font-size:11px;color:{MUTED};margin-top:4px;letter-spacing:.11em;text-transform:uppercase;}}
.sidebar-label {{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:{MUTED};margin:12px 0 8px;}}
.sidebar-status {{
  display:inline-flex;align-items:center;gap:7px;padding:6px 10px;border-radius:999px;
  background:rgba(56,217,169,.09);border:1px solid rgba(56,217,169,.2);
  color:{GREEN};font-size:11px;font-weight:700;
}}
.status-dot {{width:7px;height:7px;border-radius:50%;background:{GREEN};}}
.page-eyebrow {{font-size:11px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:{CYAN};}}
.page-title {{font-size:32px;font-weight:800;color:{TEXT};letter-spacing:-.05em;line-height:1.15;margin-top:5px;}}
.page-subtitle {{font-size:14px;color:{MUTED};margin-top:7px;}}
.header-meta {{
  display:inline-flex;align-items:center;gap:8px;padding:9px 13px;border-radius:12px;
  border:1px solid {BORDER};background:rgba(16,28,46,.8);color:{MUTED};font-size:12px;
}}
.section-kicker {{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:{MUTED};margin:12px 0 10px;}}
.kpi-grid {{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:13px;margin:20px 0 18px;}}
.kpi {{
  background:linear-gradient(145deg,rgba(20,36,58,.96),rgba(16,28,46,.96));
  border:1px solid {BORDER};border-radius:17px;padding:17px 18px;min-width:0;
  box-shadow:0 8px 26px rgba(0,0,0,.10);
}}
.kpi-head {{display:flex;align-items:center;justify-content:space-between;gap:8px;}}
.kpi-label {{font-size:12px;color:{MUTED};font-weight:600;}}
.kpi-icon {{
  width:31px;height:31px;border-radius:10px;display:flex;align-items:center;
  justify-content:center;background:rgba(66,200,245,.10);font-size:15px;
}}
.kpi-value {{font-size:27px;font-weight:800;letter-spacing:-.045em;color:{TEXT};margin-top:12px;white-space:nowrap;}}
.kpi-foot {{font-size:11px;color:{MUTED};margin-top:3px;}}
.kpi-risk {{border-color:rgba(255,113,139,.34);background:linear-gradient(145deg,rgba(255,113,139,.10),rgba(16,28,46,.96) 72%);}}
.kpi-risk .kpi-value {{color:{RED};}}
.insight-grid {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 20px;}}
.insight {{
  background:rgba(16,28,46,.78);border:1px solid {BORDER};border-radius:14px;padding:13px 15px;
}}
.insight-label {{font-size:11px;color:{MUTED};font-weight:600;}}
.insight-value {{font-size:15px;color:{TEXT};font-weight:700;margin-top:5px;}}
.insight-meta {{font-size:11px;color:{MUTED};margin-top:3px;}}
.panel-title {{font-size:15px;font-weight:700;color:{TEXT};letter-spacing:-.02em;}}
.panel-subtitle {{font-size:11px;color:{MUTED};margin-top:3px;}}
[class*="st-key-panel-"] {{
  background:rgba(16,28,46,.82);border:1px solid {BORDER};
  border-radius:17px;padding:15px 17px 8px;
}}
.group-label {{
  color:{CYAN};font-size:11px;font-weight:800;letter-spacing:.12em;
  text-transform:uppercase;margin:17px 0 8px;
}}
div[data-testid="stTabs"] button[role="tab"] {{
  color:{MUTED};font-weight:700;font-size:13px;padding:12px 17px;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
  color:{CYAN};border-bottom-color:{CYAN};
}}
.stButton > button {{
  border:1px solid {BORDER};border-radius:10px;background:{PANEL_ALT};
  color:{TEXT};font-weight:700;transition:all .18s ease;
}}
.stButton > button:hover {{border-color:{CYAN};color:{CYAN};}}
.stDownloadButton > button, button[kind="primary"] {{
  border:0;border-radius:10px;background:linear-gradient(90deg,#249DCE,{CYAN});
  color:#04111D;font-weight:800;
}}
.stDownloadButton > button:hover {{color:#04111D;border:0;filter:brightness(1.08);}}
input, textarea {{border-radius:9px!important;}}
div[data-testid="stDataFrame"] {{border:1px solid {BORDER};border-radius:12px;overflow:hidden;}}
.result-card {{
  padding:17px;border-radius:15px;border:1px solid {BORDER};background:{PANEL_ALT};margin-top:12px;
}}
.result-card.high {{border-color:rgba(255,113,139,.42);background:rgba(255,113,139,.075);}}
.result-card.medium {{border-color:rgba(246,189,96,.42);background:rgba(246,189,96,.07);}}
.result-card.low {{border-color:rgba(56,217,169,.42);background:rgba(56,217,169,.07);}}
.result-heading {{font-size:17px;font-weight:800;color:{TEXT};}}
.result-copy {{font-size:12px;line-height:1.6;color:{MUTED};margin-top:5px;}}
div[data-testid="stMetric"] {{
  background:rgba(20,36,58,.75);border:1px solid {BORDER};padding:12px 14px;border-radius:12px;
}}
div[data-testid="stMetricLabel"] {{color:{MUTED};}}
div[data-testid="stMetricValue"] {{color:{TEXT};}}
.stAlert {{border-radius:12px;}}
@media(max-width:1100px) {{
  .kpi-grid {{grid-template-columns:repeat(2,minmax(0,1fr));}}
  .insight-grid {{grid-template-columns:repeat(2,minmax(0,1fr));}}
}}
@media(max-width:650px) {{
  .block-container {{padding-left:1rem;padding-right:1rem;}}
  .page-title {{font-size:25px;}}
  .kpi-grid {{grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;}}
  .kpi {{padding:13px;}}
  .kpi-value {{font-size:22px;}}
  .insight-grid {{grid-template-columns:1fr 1fr;gap:8px;}}
}}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------- Data -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH.name}")
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"],
    )
    return data


try:
    model = load_model()
    df = load_data()
except Exception as exc:
    st.error(f"ChurnIQ could not load its files. {exc}")
    st.info(
        "Place app.py, churn_pipeline.joblib, and "
        "WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder."
    )
    st.stop()

OVERALL_RATE = (df["Churn"] == "Yes").mean() * 100


# ----------------------------- Helpers -----------------------------
def show_chart(fig, height=None):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={"displayModeBar": False, "responsive": True},
    )


def style_fig(fig, height=330, legend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=MUTED, size=11),
        showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)", title_text=""),
        hoverlabel=dict(bgcolor=PANEL_ALT, bordercolor=BORDER, font_color=TEXT),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, zeroline=False, title=None)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False, title=None)
    return fig


def churn_by(frame, column):
    return (
        frame.groupby(column, observed=True)["Churn"]
        .agg(rate=lambda s: (s == "Yes").mean() * 100, customers="size")
        .reset_index()
    )


def rate_bar(frame, column, horizontal=False, sort_by_rate=True):
    data = churn_by(frame, column)
    if sort_by_rate:
        data = data.sort_values("rate", ascending=horizontal)
    fig = px.bar(
        data,
        x="rate" if horizontal else column,
        y=column if horizontal else "rate",
        orientation="h" if horizontal else "v",
        color="rate",
        color_continuous_scale=[CYAN, PURPLE, RED],
        range_color=(0, max(60, float(data["rate"].max()))),
        custom_data=["customers"],
    )
    cat = "y" if horizontal else "x"
    val = "x" if horizontal else "y"
    fig.update_traces(
        texttemplate="%{" + val + ":.1f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "%{" + cat + "}<br>Churn rate: %{" + val + ":.1f}%"
            "<br>Customers: %{customdata[0]:,}<extra></extra>"
        ),
    )
    fig = style_fig(fig)
    if horizontal:
        fig.add_vline(x=OVERALL_RATE, line_dash="dot", line_color=MUTED, opacity=.65)
        fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=BORDER)
        fig.update_yaxes(showgrid=False)
    else:
        fig.add_hline(
            y=OVERALL_RATE, line_dash="dot", line_color=MUTED, opacity=.65,
            annotation_text=f"Overall {OVERALL_RATE:.1f}%",
            annotation_font_color=MUTED,
            annotation_position="top right",
        )
        fig.update_yaxes(ticksuffix="%")
    return fig


_panel_count = 0


def panel(title, subtitle=None):
    global _panel_count
    _panel_count += 1
    box = st.container(key=f"panel-{_panel_count}")
    box.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        box.markdown(f'<div class="panel-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    return box


def top_group(frame, column, min_size=30):
    data = churn_by(frame, column)
    eligible = data[data["customers"] >= min_size]
    if not eligible.empty:
        data = eligible
    row = data.sort_values("rate", ascending=False).iloc[0]
    return str(row[column]), float(row["rate"]), int(row["customers"])


def positive_class_info(estimator, predicted):
    """Find the churn class without assuming it is encoded as integer 1."""
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        labels = list(classes)
        for candidate in ("Yes", 1, True, "1"):
            if candidate in labels:
                idx = labels.index(candidate)
                return labels[idx], idx
        # Fall back to the second class for conventional binary classifiers.
        if len(labels) == 2:
            return labels[1], 1
    return 1, 1


def prediction_label(value):
    return str(value).strip().lower() in {"yes", "1", "true", "churn", "churned"}


def safe_percent(value):
    return f"{value:.1f}%" if pd.notna(value) else "N/A"


# ----------------------------- Sidebar -----------------------------
contracts = sorted(df["Contract"].dropna().unique().tolist())
internets = sorted(df["InternetService"].dropna().unique().tolist())
T_MIN, T_MAX = int(df["tenure"].min()), int(df["tenure"].max())
M_MIN = float(math.floor(df["MonthlyCharges"].min()))
M_MAX = float(math.ceil(df["MonthlyCharges"].max()))

FILTER_DEFAULTS = {
    "f_contract": [],
    "f_internet": [],
    "f_tenure": (T_MIN, T_MAX),
    "f_monthly": (M_MIN, M_MAX),
}
for key, value in FILTER_DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_filters():
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value


active_filters = sum(
    1 for key, default in FILTER_DEFAULTS.items()
    if tuple(st.session_state[key]) != tuple(default)
)

with st.sidebar:
    st.markdown(
        f"""
        <div class="side-brand">
          <div class="brand-mark">C</div>
          <div>
            <div class="brand-name">ChurnIQ</div>
            <div class="brand-caption">Customer intelligence</div>
          </div>
        </div>
        <div class="sidebar-status"><span class="status-dot"></span> Analytics workspace</div>
        <div class="sidebar-label">Dashboard filters</div>
        """,
        unsafe_allow_html=True,
    )
    st.pills("Contract type", contracts, selection_mode="multi", key="f_contract")
    st.pills("Internet service", internets, selection_mode="multi", key="f_internet")
    st.caption("Leave a category unselected to include all values.")
    st.slider("Customer tenure", T_MIN, T_MAX, key="f_tenure")
    st.slider("Monthly charges ($)", M_MIN, M_MAX, step=1.0, key="f_monthly")
    st.button("↺  Reset all filters", on_click=reset_filters, use_container_width=True)
    st.divider()
    st.markdown('<div class="sidebar-label">About this project</div>', unsafe_allow_html=True)
    st.caption("Telco customer churn analytics and machine-learning prediction.")
    st.caption(f"Dataset · {len(df):,} customer records")

selected_contracts = st.session_state["f_contract"] or contracts
selected_internet = st.session_state["f_internet"] or internets

fdf = df[
    df["Contract"].isin(selected_contracts)
    & df["InternetService"].isin(selected_internet)
    & df["tenure"].between(*st.session_state["f_tenure"])
    & df["MonthlyCharges"].between(*st.session_state["f_monthly"])
].copy()

# ----------------------------- Header -----------------------------
header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown('<div class="page-eyebrow">TELCO CUSTOMER INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Customer retention overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Understand churn patterns, explore customer segments, and estimate individual churn risk.</div>',
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        f'<div class="header-meta">◉ &nbsp; {len(fdf):,} / {len(df):,} customers in view</div>',
        unsafe_allow_html=True,
    )

if fdf.empty:
    st.warning("No customers match the selected filters. Reset the filters to restore the full dataset.")
    st.stop()

# ----------------------------- KPI row -----------------------------
total_customers = len(fdf)
churned_customers = int((fdf["Churn"] == "Yes").sum())
retained_customers = total_customers - churned_customers
churn_rate = churned_customers / total_customers * 100
rate_difference = churn_rate - OVERALL_RATE

if abs(rate_difference) < .05:
    comparison = "In line with overall dataset"
elif rate_difference > 0:
    comparison = f"{rate_difference:.1f} pts above dataset average"
else:
    comparison = f"{abs(rate_difference):.1f} pts below dataset average"

kpis = [
    ("TOTAL CUSTOMERS", f"{total_customers:,}", "Customers in current view", "👥", ""),
    ("CHURNED CUSTOMERS", f"{churned_customers:,}", f"{retained_customers:,} customers retained", "↗", ""),
    ("CHURN RATE", f"{churn_rate:.1f}%", comparison, "◎", "risk"),
    ("AVG. MONTHLY BILL", f"${fdf['MonthlyCharges'].mean():,.2f}", "Average recurring charge", "$", ""),
    ("AVG. TENURE", f"{fdf['tenure'].mean():.1f} mo", "Customer relationship length", "◷", ""),
]
kpi_html = '<div class="kpi-grid">'
for label, value, foot, symbol, extra in kpis:
    kpi_html += (
        f'<div class="kpi {("kpi-risk" if extra else "")}">'
        f'<div class="kpi-head"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-icon">{symbol}</div></div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-foot">{foot}</div></div>'
    )
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

# ----------------------------- Automatic insights -----------------------------
st.markdown('<div class="section-kicker">Segment signals · Current filter selection</div>', unsafe_allow_html=True)
insight_items = [
    ("Highest-risk contract", "Contract"),
    ("Highest-risk tenure group", "TenureGroup"),
    ("Highest-risk payment method", "PaymentMethod"),
    ("Highest-risk internet service", "InternetService"),
]
insight_html = '<div class="insight-grid">'
for label, column in insight_items:
    name, segment_rate, count = top_group(fdf, column)
    insight_html += (
        f'<div class="insight"><div class="insight-label">{label}</div>'
        f'<div class="insight-value">{name}</div>'
        f'<div class="insight-meta">{segment_rate:.1f}% churn · {count:,} customers</div></div>'
    )
insight_html += "</div>"
st.markdown(insight_html, unsafe_allow_html=True)

# ----------------------------- Navigation -----------------------------
tab_overview, tab_deep, tab_predict, tab_model = st.tabs(
    ["◫  Overview", "⌁  Deep analytics", "✳  Churn prediction", "⚙  Model information"]
)

# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:
    st.markdown('<div class="section-kicker">Churn performance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([.9, 1.35], gap="large")

    with c1:
        with panel("Customer outcome", "Distribution of retained and churned customers"):
            split = fdf["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0)
            donut = go.Figure(
                go.Pie(
                    labels=["Retained", "Churned"],
                    values=split.values,
                    hole=.76,
                    sort=False,
                    marker=dict(colors=[CYAN, RED], line=dict(color=PANEL, width=3)),
                    textinfo="none",
                    hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
                )
            )
            donut.add_annotation(
                text=f"<b style='font-size:29px;color:{TEXT}'>{churn_rate:.1f}%</b>"
                     f"<br><span style='font-size:11px;color:{MUTED}'>CHURN RATE</span>",
                showarrow=False,
            )
            show_chart(style_fig(donut, 320, True))

    with c2:
        with panel("Churn by contract", "Monthly contracts show a different retention pattern from longer commitments"):
            show_chart(rate_bar(fdf, "Contract"), 320)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        with panel("Churn by customer tenure", "How churn varies with relationship length"):
            show_chart(rate_bar(fdf, "TenureGroup", sort_by_rate=False), 310)
    with c4:
        with panel("Monthly charges by outcome", "Distribution of monthly charges for retained and churned customers"):
            hist = px.histogram(
                fdf, x="MonthlyCharges", color="Churn", nbins=32,
                barmode="overlay", opacity=.76,
                color_discrete_map={"No": CYAN, "Yes": RED},
                category_orders={"Churn": ["No", "Yes"]},
            )
            hist.update_traces(marker_line_width=0)
            hist = style_fig(hist, 310, True)
            hist.update_xaxes(tickprefix="$")
            show_chart(hist)

    st.caption("Interpretation note: these charts show associations in the dataset; they do not establish that a feature causes churn.")

# ============================================================
# DEEP ANALYTICS
# ============================================================
with tab_deep:
    st.markdown('<div class="section-kicker">Explore customer segments</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="large")
    with d1:
        with panel("Internet service", "Churn rate by internet service type"):
            show_chart(rate_bar(fdf, "InternetService"), 315)
    with d2:
        with panel("Payment method", "Dotted line represents the full-dataset churn rate"):
            show_chart(rate_bar(fdf, "PaymentMethod", horizontal=True), 315)

    h1, h2 = st.columns([1, 1.25], gap="large")
    with h1:
        with panel("Contract × tenure", "Churn rate (%) across contract and tenure segments"):
            pivot = (
                fdf.groupby(["Contract", "TenureGroup"], observed=True)["Churn"]
                .apply(lambda values: (values == "Yes").mean() * 100)
                .unstack()
            )
            pivot = pivot.reindex(
                [c for c in ["Month-to-month", "One year", "Two year"] if c in pivot.index]
            )
            pivot.columns = pivot.columns.astype(str)
            heat = px.imshow(
                pivot, aspect="auto", text_auto=".0f",
                color_continuous_scale=[PANEL_ALT, PURPLE, RED], zmin=0, zmax=60,
            )
            heat.update_traces(
                texttemplate="%{z:.0f}%", textfont=dict(color=TEXT, size=12),
                xgap=4, ygap=4,
                hovertemplate="%{y} · %{x}<br>Churn rate: %{z:.1f}%<extra></extra>",
            )
            heat = style_fig(heat, 310)
            heat.update_xaxes(side="bottom", showgrid=False)
            heat.update_yaxes(showgrid=False)
            show_chart(heat)
    with h2:
        with panel("Tenure vs. monthly charges", "A sample of up to 1,500 customers; each point represents one customer"):
            sample = fdf.sample(min(len(fdf), 1500), random_state=42)
            scatter = px.scatter(
                sample, x="tenure", y="MonthlyCharges", color="Churn",
                opacity=.72, color_discrete_map={"No": CYAN, "Yes": RED},
                category_orders={"Churn": ["No", "Yes"]},
            )
            scatter.update_traces(marker=dict(size=7, line=dict(width=0)))
            scatter = style_fig(scatter, 310, True)
            scatter.update_xaxes(title="Tenure (months)")
            scatter.update_yaxes(title="Monthly charges ($)", tickprefix="$")
            show_chart(scatter)

    st.markdown('<div class="section-kicker">Customer-level data</div>', unsafe_allow_html=True)
    with panel("Customer records", "Search, inspect, and export customers included in the current filter selection"):
        search_col, info_col = st.columns([1.3, 1])
        with search_col:
            search = st.text_input(
                "Search customer ID",
                placeholder="e.g. 7590-VHVEG",
                label_visibility="collapsed",
                key="customer_search",
            )
        with info_col:
            st.caption("Showing up to 200 rows in the preview. CSV export includes all matching rows.")
        table = fdf.drop(columns=["TenureGroup"]).copy()
        if search:
            table = table[table["customerID"].astype(str).str.contains(search, case=False, na=False)]
        st.dataframe(
            table.head(200),
            use_container_width=True,
            hide_index=True,
            height=350,
            column_config={
                "MonthlyCharges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
                "TotalCharges": st.column_config.NumberColumn("Total charges", format="$%.2f"),
            },
        )
        st.caption(f"{len(table):,} matching customer records")
        st.download_button(
            "↓  Export matching customers (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="churniq_customer_export.csv",
            mime="text/csv",
            type="primary",
            use_container_width=False,
        )

# ============================================================
# CHURN PREDICTION
# ============================================================
DEFAULTS = {
    "p_contract": "Month-to-month", "p_tenure": 12, "p_paperless": "Yes",
    "p_monthly": 70.0, "p_manual_total": False, "p_total": 840.0,
    "p_payment": "Electronic check", "p_gender": "Female", "p_senior": 0,
    "p_partner": "No", "p_dependents": "No", "p_phone": "Yes", "p_multiple": "No",
    "p_internet": "Fiber optic", "p_security": "No", "p_backup": "No",
    "p_protection": "No", "p_support": "No", "p_tv": "No", "p_movies": "No",
}
PRESETS = {
    "risky": {
        **DEFAULTS, "p_tenure": 3, "p_monthly": 85.0,
        "p_internet": "Fiber optic", "p_payment": "Electronic check",
        "p_contract": "Month-to-month",
    },
    "loyal": {
        **DEFAULTS, "p_contract": "Two year", "p_tenure": 60,
        "p_monthly": 65.0, "p_paperless": "No",
        "p_payment": "Credit card (automatic)", "p_partner": "Yes",
        "p_dependents": "Yes", "p_internet": "DSL",
        "p_security": "Yes", "p_backup": "Yes", "p_protection": "Yes",
        "p_support": "Yes",
    },
    "reset": DEFAULTS,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def apply_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


with tab_predict:
    st.markdown('<div class="section-kicker">Individual customer assessment</div>', unsafe_allow_html=True)
    left, right = st.columns([1.45, 1], gap="large")

    with left:
        with panel("Customer profile", "Enter a customer profile or use a sample scenario to test the model."):
            st.markdown('<div class="group-label">Sample scenarios</div>', unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            b1.button("Higher risk", on_click=apply_preset, args=("risky",), use_container_width=True)
            b2.button("Lower risk", on_click=apply_preset, args=("loyal",), use_container_width=True)
            b3.button("Reset", on_click=apply_preset, args=("reset",), use_container_width=True)

            st.markdown('<div class="group-label">Account details</div>', unsafe_allow_html=True)
            a1, a2, a3 = st.columns(3)
            contract = a1.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="p_contract")
            tenure = a2.number_input("Tenure (months)", 0, 72, key="p_tenure")
            paperless = a3.selectbox("Paperless billing", ["Yes", "No"], key="p_paperless")

            m1, m2, m3 = st.columns(3)
            monthly = m1.number_input("Monthly charges ($)", min_value=0.0, step=5.0, key="p_monthly")
            manual_total = m2.toggle("Set total manually", key="p_manual_total")
            if manual_total:
                total = m3.number_input("Total charges ($)", min_value=0.0, step=50.0, key="p_total")
            else:
                total = float(monthly) * int(tenure)
                m3.metric("Estimated total", f"${total:,.2f}")
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="p_payment",
            )

            st.markdown('<div class="group-label">Customer details</div>', unsafe_allow_html=True)
            p1, p2, p3, p4 = st.columns(4)
            gender = p1.selectbox("Gender", ["Female", "Male"], key="p_gender")
            senior = p2.selectbox("Senior citizen", [0, 1], key="p_senior")
            partner = p3.selectbox("Has partner", ["Yes", "No"], key="p_partner")
            dependents = p4.selectbox("Dependents", ["Yes", "No"], key="p_dependents")

            st.markdown('<div class="group-label">Subscribed services</div>', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            phone = s1.selectbox("Phone service", ["Yes", "No"], key="p_phone")
            internet = s2.selectbox("Internet service", ["DSL", "Fiber optic", "No"], key="p_internet")

            if phone == "No":
                st.session_state["p_multiple"] = "No phone service"
                multi_options = ["No phone service"]
            else:
                if st.session_state["p_multiple"] == "No phone service":
                    st.session_state["p_multiple"] = "No"
                multi_options = ["No", "Yes"]

            internet_keys = ["p_security", "p_backup", "p_protection", "p_support", "p_tv", "p_movies"]
            if internet == "No":
                for key in internet_keys:
                    st.session_state[key] = "No internet service"
                internet_options = ["No internet service"]
            else:
                for key in internet_keys:
                    if st.session_state[key] == "No internet service":
                        st.session_state[key] = "No"
                internet_options = ["Yes", "No"]

            multiple = st.selectbox(
                "Multiple lines", multi_options, key="p_multiple", disabled=phone == "No"
            )
            u1, u2, u3 = st.columns(3)
            disabled = internet == "No"
            security = u1.selectbox("Online security", internet_options, key="p_security", disabled=disabled)
            backup = u2.selectbox("Online backup", internet_options, key="p_backup", disabled=disabled)
            protection = u3.selectbox("Device protection", internet_options, key="p_protection", disabled=disabled)
            v1, v2, v3 = st.columns(3)
            support = v1.selectbox("Tech support", internet_options, key="p_support", disabled=disabled)
            streaming_tv = v2.selectbox("Streaming TV", internet_options, key="p_tv", disabled=disabled)
            streaming_movies = v3.selectbox("Streaming movies", internet_options, key="p_movies", disabled=disabled)

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    try:
        predicted_value = model.predict(customer)[0]
        probabilities = model.predict_proba(customer)[0]
        positive_label, positive_index = positive_class_info(model, predicted_value)
        probability = float(probabilities[positive_index])
        predicted_churn = prediction_label(predicted_value)
    except Exception as exc:
        probability = None
        predicted_churn = None
        st.error(f"Prediction failed: {exc}")
        st.info("Check that the input columns and categories match those used when training churn_pipeline.joblib.")

    with right:
        with panel("Model assessment", "Estimated probability that this customer will churn"):
            if probability is not None:
                if probability >= .60:
                    risk_level, risk_class, risk_color = "High risk", "high", RED
                    recommendation = "Prioritize this customer for a retention conversation. Review service issues and consider a relevant retention offer."
                    verdict = "Customer may be at risk of leaving"
                elif probability >= .35:
                    risk_level, risk_class, risk_color = "Moderate risk", "medium", AMBER
                    recommendation = "Consider a proactive check-in and monitor engagement before offering an incentive."
                    verdict = "Customer shows some churn signals"
                else:
                    risk_level, risk_class, risk_color = "Low risk", "low", GREEN
                    recommendation = "No immediate intervention is indicated by this model. Continue normal customer engagement."
                    verdict = "Customer is less likely to churn"

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number=dict(suffix="%", font=dict(size=43, color=TEXT, family="DM Sans")),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickvals=[0, 35, 60, 100], tickcolor=MUTED),
                        bar=dict(color=risk_color, thickness=.25),
                        bgcolor=PANEL_ALT,
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 35], color="rgba(56,217,169,.13)"),
                            dict(range=[35, 60], color="rgba(246,189,96,.13)"),
                            dict(range=[60, 100], color="rgba(255,113,139,.13)"),
                        ],
                    ),
                ))
                show_chart(style_fig(gauge, 245))

                st.markdown(
                    f"""
                    <div class="result-card {risk_class}">
                      <div class="result-heading">{verdict}</div>
                      <div class="result-copy"><b style="color:{TEXT}">{risk_level}</b> · Model probability: <b style="color:{TEXT}">{probability:.1%}</b></div>
                      <div class="result-copy">{recommendation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="group-label">Context and comparison</div>', unsafe_allow_html=True)
                similar = df[
                    (df["Contract"] == contract)
                    & (df["InternetService"] == internet)
                ]
                similar_rate = (similar["Churn"] == "Yes").mean() * 100 if len(similar) else float("nan")
                difference = probability * 100 - OVERALL_RATE
                q1, q2 = st.columns(2)
                q1.metric("Dataset churn rate", f"{OVERALL_RATE:.1f}%")
                q2.metric("Similar segment rate", safe_percent(similar_rate))
                st.caption(
                    f"The model estimate is {abs(difference):.1f} percentage points "
                    f"{'above' if difference >= 0 else 'below'} the dataset churn rate."
                )
                st.caption("Risk bands are illustrative decision thresholds, not calibrated business rules. A prediction is not a guarantee.")
            else:
                st.info("Complete the customer profile to view a model estimate.")

# ============================================================
# MODEL INFORMATION
# ============================================================
with tab_model:
    st.markdown('<div class="section-kicker">Transparency and methodology</div>', unsafe_allow_html=True)
    m1, m2 = st.columns([1.1, 1], gap="large")
    with m1:
        with panel("Model overview", "The trained pipeline loaded by this application"):
            st.markdown("**Model artifact**")
            st.code(MODEL_PATH.name, language="text")
            st.markdown("**Prediction target**")
            st.write("Telco customer churn — whether a customer is likely to leave.")
            st.markdown("**Inference workflow**")
            st.write("Customer inputs → fitted preprocessing pipeline → trained classifier → churn probability.")
            st.markdown("**Input data**")
            st.write(f"Original Telco dataset · {len(df):,} customer records · {len(df.columns)} columns including the derived tenure group.")
    with m2:
        with panel("Dataset profile", "Basic quality and target-distribution checks"):
            profile = pd.DataFrame({
                "Metric": ["Customer records", "Original features", "Missing values (all columns)", "Duplicate rows", "Churned customers", "Retained customers"],
                "Value": [
                    f"{len(df):,}",
                    f"{len(df.columns) - 1:,}",
                    f"{int(df.isna().sum().sum()):,}",
                    f"{int(df.duplicated().sum()):,}",
                    f"{int((df['Churn'] == 'Yes').sum()):,}",
                    f"{int((df['Churn'] == 'No').sum()):,}",
                ],
            })
            st.dataframe(profile, hide_index=True, use_container_width=True)
            target_fig = px.pie(
                names=["Retained", "Churned"],
                values=[int((df["Churn"] == "No").sum()), int((df["Churn"] == "Yes").sum())],
                hole=.68,
                color_discrete_sequence=[CYAN, RED],
            )
            target_fig.update_traces(textinfo="percent", hovertemplate="%{label}: %{value:,}<extra></extra>")
            show_chart(style_fig(target_fig, 230, True))
    st.warning(
        "Important: the dashboard is a portfolio demonstration. Before operational use, "
        "validate probability calibration, evaluate false positives/false negatives, "
        "review fairness, and define retention actions with business stakeholders."
    )

st.markdown("---")
st.markdown(
    f'<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:{MUTED};font-size:11px;">'
    f'<span><b style="color:{TEXT}">ChurnIQ</b> · Telco Customer Churn Analytics</span>'
    f'<span>Built with Streamlit · Pandas · Plotly · Scikit-learn</span></div>',
    unsafe_allow_html=True,
)

