# Superstore Sales Insights Dashboard — Built by Shubh Kumar
# Run: streamlit run app.py
# Notes:
# - Works with the common "Global Superstore" / "Sample - Superstore" datasets.
# - If you don't have the CSV, the app can generate a small synthetic demo dataset.
# - Expected columns (case-insensitive):
#   Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Customer Name,
#   Segment, Country, City, State, Postal Code, Region, Product ID, Category,
#   Sub-Category, Product Name, Sales, Quantity, Discount, Profit

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

# ---------------------- App Config ----------------------
st.set_page_config(
    page_title="Superstore Sales Insights Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- Styles --------------------------
HIDE_DECOR = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .small {font-size:0.85rem; opacity:0.8}
    </style>
"""
st.markdown(HIDE_DECOR, unsafe_allow_html=True)

# ---------------------- Helpers -------------------------
REQUIRED_COLS = [
    "Order ID","Order Date","Ship Date","Ship Mode","Customer ID","Customer Name",
    "Segment","Country","City","State","Postal Code","Region","Product ID",
    "Category","Sub-Category","Product Name","Sales","Quantity","Discount","Profit"
]

COL_ALIASES = {c.lower(): c for c in REQUIRED_COLS}

@st.cache_data(show_spinner=False)
def load_csv(file: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(file, encoding_errors='ignore')
    return df

@st.cache_data(show_spinner=False)
def coerce_superstore_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize columns (case-insensitive matches to REQUIRED_COLS)
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COL_ALIASES:
            rename_map[col] = COL_ALIASES[key]
    df = df.rename(columns=rename_map).copy()

    # Add missing columns if any
    for col in REQUIRED_COLS:
        if col not in df.columns:
            # Create safe defaults
            if col in ["Sales","Profit","Discount"]:
                df[col] = 0.0
            elif col in ["Quantity","Postal Code"]:
                df[col] = 0
            else:
                df[col] = np.nan

    # Parse dates
    for dcol in ["Order Date", "Ship Date"]:
        df[dcol] = pd.to_datetime(df[dcol], errors='coerce')

    # Clean numerics
    for ncol in ["Sales","Profit","Discount"]:
        df[ncol] = pd.to_numeric(df[ncol], errors='coerce')

    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')

    # Enrich
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period('M').dt.to_timestamp()
    df["Month Name"] = df["Order Date"].dt.strftime('%b %Y')

    # Guard against completely invalid/missing dates
    df = df.dropna(subset=["Order Date"]).copy()

    return df

@st.cache_data(show_spinner=False)
def sample_data(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = np.datetime64('2015-01-01')
    end = np.datetime64('2018-12-31')
    dates = rng.integers((end - start).astype(int), size=n) + start
    categories = ["Furniture", "Office Supplies", "Technology"]
    subcats = {
        "Furniture": ["Chairs","Tables","Bookcases"],
        "Office Supplies": ["Binders","Paper","Storage"],
        "Technology": ["Phones","Accessories","Machines"]
    }
    regions = ["East","West","Central","South"]
    states = ["California","New York","Texas","Washington","Florida","Illinois","Pennsylvania","Ohio"]

    cat = rng.choice(categories, n)
    sub = [rng.choice(subcats[c]) for c in cat]
    reg = rng.choice(regions, n)
    stt = rng.choice(states, n)

    qty = rng.integers(1, 6, size=n)
    base_price = rng.normal(120, 60, size=n).clip(5)
    discount = np.clip(rng.normal(0.1, 0.08, size=n), 0, 0.5)
    sales = (base_price * qty) * (1 - discount)
    cost = base_price * qty * rng.uniform(0.6, 0.9, size=n)
    profit = sales - cost

    df = pd.DataFrame({
        "Order ID": [f"ORD-{i:06d}" for i in range(n)],
        "Order Date": pd.to_datetime(dates.astype(str)),
        "Ship Date": pd.to_datetime(dates.astype(str)) + pd.to_timedelta(rng.integers(1,9,size=n), unit='D'),
        "Ship Mode": rng.choice(["Second Class","Standard Class","First Class","Same Day"], n),
        "Customer ID": [f"C-{rng.integers(1000,9999)}" for _ in range(n)],
        "Customer Name": [f"Customer {i}" for i in range(n)],
        "Segment": rng.choice(["Consumer","Corporate","Home Office"], n),
        "Country": "United States",
        "City": "",
        "State": stt,
        "Postal Code": 0,
        "Region": reg,
        "Product ID": [f"P-{rng.integers(10000,99999)}" for _ in range(n)],
        "Category": cat,
        "Sub-Category": sub,
        "Product Name": [f"Product {i}" for i in range(n)],
        "Sales": sales.round(2),
        "Quantity": qty,
        "Discount": discount.round(2),
        "Profit": profit.round(2)
    })

    return coerce_superstore_schema(df)

@st.cache_data(show_spinner=False)
def apply_filters(df: pd.DataFrame,
                  date_range,
                  regions: list,
                  segments: list,
                  categories: list,
                  subcats: list) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if date_range is not None and len(date_range) == 2:
        start, end = date_range
        if start is not None:
            mask &= df["Order Date"].ge(pd.to_datetime(start))
        if end is not None:
            mask &= df["Order Date"].le(pd.to_datetime(end))
    if regions:
        mask &= df["Region"].isin(regions)
    if segments:
        mask &= df["Segment"].isin(segments)
    if categories:
        mask &= df["Category"].isin(categories)
    if subcats:
        mask &= df["Sub-Category"].isin(subcats)

    return df.loc[mask].copy()

# ---------------------- Sidebar -------------------------
st.sidebar.title("📦 Superstore Dashboard")

with st.sidebar.expander("1) Load Data", expanded=True):
    uploaded = st.file_uploader("Upload Superstore CSV", type=["csv"])
    use_demo = st.toggle("Use demo dataset", value=True if uploaded is None else False)

if uploaded is not None:
    raw = load_csv(uploaded)
else:
    raw = sample_data() if use_demo else pd.DataFrame()

if raw.empty:
    st.warning("Upload a CSV or enable the demo dataset from the sidebar.")
    st.stop()

# Standardize & enrich
ss = coerce_superstore_schema(raw)

# Sidebar Filters
with st.sidebar.expander("2) Filters", expanded=True):
    min_date = ss["Order Date"].min()
    max_date = ss["Order Date"].max()
    date_range = st.date_input("Date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())

    cols = st.columns(2)
    with cols[0]:
        regions = st.multiselect("Region", options=sorted(ss["Region"].dropna().unique()), default=[])
        categories = st.multiselect("Category", options=sorted(ss["Category"].dropna().unique()), default=[])
    with cols[1]:
        segments = st.multiselect("Segment", options=sorted(ss["Segment"].dropna().unique()), default=[])
        subcats = st.multiselect("Sub-Category", options=sorted(ss["Sub-Category"].dropna().unique()), default=[])

filtered = apply_filters(ss, date_range, regions, segments, categories, subcats)

# ---------------------- KPIs ----------------------------
@st.cache_data(show_spinner=False)
def compute_kpis(df: pd.DataFrame):
    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    orders = df["Order ID"].nunique()
    qty = int(df["Quantity"].sum()) if "Quantity" in df.columns else None
    aov = total_sales / max(orders, 1)
    margin = (total_profit / total_sales) * 100 if total_sales else 0

    # Prior period for simple deltas
    if df["Month"].nunique() >= 2:
        latest_month = df["Month"].max()
        prev_month = (latest_month.to_period('M') - 1).to_timestamp()
        cur_sales = df.loc[df["Month"] == latest_month, "Sales"].sum()
        prev_sales = df.loc[df["Month"] == prev_month, "Sales"].sum()
        delta_sales = cur_sales - prev_sales
        delta_pct = (delta_sales / prev_sales * 100) if prev_sales else 0
    else:
        delta_sales = 0
        delta_pct = 0

    return {
        "sales": total_sales,
        "profit": total_profit,
        "orders": orders,
        "qty": qty,
        "aov": aov,
        "margin": margin,
        "delta_sales": delta_sales,
        "delta_pct": delta_pct,
    }

kpis = compute_kpis(filtered)

st.title("Superstore Sales Insights Dashboard")
st.caption("Interactive KPIs to identify top-selling products, profitable regions, and discount impacts — Streamlit + Plotly | Built by Shubh Kumar")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total Sales", f"$ {kpis['sales']:,.0f}", delta=f"{kpis['delta_pct']:.1f}% vs. prev mo")
with m2:
    st.metric("Total Profit", f"$ {kpis['profit']:,.0f}")
with m3:
    st.metric("Orders", f"{kpis['orders']:,}")
with m4:
    st.metric("Avg Order Value", f"$ {kpis['aov']:,.0f}")
with m5:
    st.metric("Profit Margin", f"{kpis['margin']:.1f}%")

st.divider()

# ---------------------- Time Series ---------------------
@st.cache_data(show_spinner=False)
def sales_profit_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    ts = df.groupby("Month", as_index=False)[["Sales","Profit"]].sum().sort_values("Month")
    ts_long = ts.melt(id_vars="Month", value_vars=["Sales","Profit"], var_name="Metric", value_name="Amount")
    return ts_long

ts_long = sales_profit_timeseries(filtered)

fig_ts = px.line(
    ts_long, x="Month", y="Amount", color="Metric",
    title="Sales & Profit Over Time (Monthly)", markers=True
)
st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------- Category View -------------------
cc1, cc2 = st.columns([1,1])

with cc1:
    cat_sales = filtered.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    fig_cat = px.bar(cat_sales, x="Category", y="Sales", title="Sales by Category", text_auto=True)
    st.plotly_chart(fig_cat, use_container_width=True)

with cc2:
    sub_profit = filtered.groupby(["Category","Sub-Category"], as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
    fig_sub = px.treemap(sub_profit, path=["Category","Sub-Category"], values="Profit", title="Profit by Sub-Category (Treemap)")
    st.plotly_chart(fig_sub, use_container_width=True)

# ---------------------- Top Products --------------------
left, right = st.columns([1,1])

with left:
    prod_sales = (
        filtered.groupby(["Product Name"], as_index=False)
                .agg(Sales=("Sales","sum"), Profit=("Profit","sum"))
                .sort_values("Sales", ascending=False)
                .head(10)
    )
    fig_topprod = px.bar(prod_sales, x="Product Name", y="Sales", title="Top 10 Products by Sales")
    fig_topprod.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_topprod, use_container_width=True)

with right:
    # Discount impact: average profit by discount bucket
    bins = [-0.01, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    labels = ["0%","0-5%","5-10%","10-20%","20-30%","30-50%",">=50%"]
    tmp = filtered.copy()
    tmp["Discount Bucket"] = pd.cut(tmp["Discount"].fillna(0.0), bins=bins, labels=labels)
    disc_agg = tmp.groupby("Discount Bucket", as_index=False)["Profit"].mean().dropna()
    fig_disc = px.bar(disc_agg, x="Discount Bucket", y="Profit", title="Avg Profit vs Discount Bucket")
    st.plotly_chart(fig_disc, use_container_width=True)

# ---------------------- Geography (State) ----------------
if filtered["State"].notna().any():
    st.subheader("Profit by State")
    state_profit = filtered.groupby("State", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
    fig_state = px.bar(state_profit.head(25), x="State", y="Profit", title="Top States by Profit")
    fig_state.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_state, use_container_width=True)

# ---------------------- Customer RFM --------------------
st.subheader("Customer RFM (Recency–Frequency–Monetary)")

@st.cache_data(show_spinner=False)
def compute_rfm(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    snapshot = df["Order Date"].max()
    rfm = (
        df.groupby(["Customer ID","Customer Name"], as_index=False)
          .agg(
              RecencyDays=("Order Date", lambda s: (snapshot - s.max()).days),
              Frequency=("Order ID","nunique"),
              Monetary=("Sales","sum")
          )
    )
    # Score each 1-5 (5 is best)
    rfm['R_Score'] = pd.qcut(-rfm['RecencyDays'], 5, labels=[1,2,3,4,5])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].astype(int).sum(axis=1)
    rfm['Segment'] = pd.cut(
        rfm['RFM_Score'],
        bins=[0,6,10,15],
        labels=["Bronze","Silver","Gold"],
        include_lowest=True
    )
    return rfm.sort_values('RFM_Score', ascending=False)

rfm = compute_rfm(filtered)

c1, c2 = st.columns([2,1])
with c1:
    st.dataframe(rfm, use_container_width=True, height=360)
with c2:
    top_customers = rfm.head(10)
    fig_rfm = px.bar(top_customers, x="Customer Name", y="Monetary", title="Top Customers (Monetary)")
    fig_rfm.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_rfm, use_container_width=True)

# ---------------------- Downloads -----------------------
@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

st.download_button("Download Filtered Data (CSV)", data=to_csv_bytes(filtered), file_name="filtered_superstore.csv", mime="text/csv")
st.download_button("Download RFM (CSV)", data=to_csv_bytes(rfm), file_name="rfm_superstore.csv", mime="text/csv")

# ---------------------- Details/Notes -------------------
with st.expander("About this dashboard"):
    st.markdown(
        """
        **Purpose**: Analyze Superstore sales, profit, customers, and discounts to generate actionable insights.  
        **How to use**: Load your CSV (or use demo data), apply filters on the left, explore KPIs and charts, and export tables.

        **Key Insights you can derive**
        - Trend of Sales and Profit over time (seasonality, growth/decline).
        - Most profitable Categories/Sub-Categories (Treemap) and products.
        - Discount buckets that erode or support profitability.
        - State/Region performance to focus on best/worst locations.
        - Customer RFM to target Gold/Silver/Bronze segments.
        """
    )


