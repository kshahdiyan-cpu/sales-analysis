import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Sales Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

MODELS = [
    "Linear Regression (trend + peak season)",
    "Ridge (trend + monthly seasonality)",
    "Hybrid (linear trend + Random Forest)",
]


# ---------- Data ----------
@st.cache_data
def load_data():
    return pd.read_csv("data/sales_clean.csv", parse_dates=["order_date"])


@st.cache_data
def build_rfm(df, snapshot):
    rfm = df.groupby("customer_id").agg(
        Recency=("order_date", lambda x: (snapshot - x.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("sales", "sum"),
    )
    if len(rfm) < 3:  # too few customers to cluster
        rfm["Segment"] = "Regular"
        return rfm.reset_index()
    scaled = StandardScaler().fit_transform(rfm)
    rfm["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(scaled)
    means = rfm.groupby("cluster")[["Recency", "Monetary"]].mean()
    at_risk = means["Recency"].idxmax()
    champions = means.drop(at_risk)["Monetary"].idxmax()
    names = {c: "Regular" for c in means.index}
    names[at_risk] = "At-Risk"
    names[champions] = "Champions"
    rfm["Segment"] = rfm["cluster"].map(names)
    return rfm.reset_index()


# ---------- Forecasting helpers ----------
def _peak_feats(df):
    m = df["order_date"].dt.month
    return pd.DataFrame({"t": df["t"].values, "peak": m.isin([11, 12]).astype(int).values})


def _month_feats(df):
    m = df["order_date"].dt.month
    X = pd.DataFrame({"t": df["t"].values})
    for i in range(2, 13):
        X[f"m{i}"] = (m == i).astype(int).values
    return X


def _rf_feats(df):
    m = df["order_date"].dt.month
    return pd.DataFrame({"month": m.values, "peak": m.isin([11, 12]).astype(int).values})


def fit_predict(name, train, future):
    """Fit on train, return (fitted_values, future_predictions)."""
    y = train["sales"].values
    if name == MODELS[0]:
        model = LinearRegression().fit(_peak_feats(train), y)
        return model.predict(_peak_feats(train)), model.predict(_peak_feats(future))
    if name == MODELS[1]:
        model = Ridge(alpha=1.0).fit(_month_feats(train), y)
        return model.predict(_month_feats(train)), model.predict(_month_feats(future))
    # Hybrid: linear trend (can extrapolate) + Random Forest on the seasonal residuals
    lin = LinearRegression().fit(train[["t"]], y)
    resid = y - lin.predict(train[["t"]])
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
    rf.fit(_rf_feats(train), resid)
    fitted = lin.predict(train[["t"]]) + rf.predict(_rf_feats(train))
    pred = lin.predict(future[["t"]]) + rf.predict(_rf_feats(future))
    return fitted, pred


@st.cache_data
def backtest(monthly, n_test=6):
    train, test = monthly.iloc[:-n_test], monthly.iloc[-n_test:]
    y = test["sales"].values
    rows = {}
    for name in MODELS:
        _, p = fit_predict(name, train, test)
        err = y - p
        safe_y = np.where(y == 0, np.nan, y)
        rows[name] = {
            "MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "MAPE %": np.nanmean(np.abs(err) / safe_y) * 100,
        }
    return pd.DataFrame(rows).T


def monthly_series(frame, index=None):
    s = frame.set_index("order_date").resample("MS")["sales"].sum()
    if index is not None:
        s = s.reindex(index, fill_value=0)
    out = s.reset_index()
    out.columns = ["order_date", "sales"]
    out["t"] = np.arange(len(out))
    return out


df = load_data()

# ---------- Sidebar filters ----------
st.sidebar.title("🔎 Filters")
month_opts = [p.strftime("%b %Y") for p in pd.period_range(df.order_date.min(), df.order_date.max(), freq="M")]
start_lbl, end_lbl = st.sidebar.select_slider("Period (drag both ends)", options=month_opts,
                                              value=(month_opts[0], month_opts[-1]))
regions = st.sidebar.multiselect("Region", sorted(df.region.unique()), sorted(df.region.unique()))
cats = st.sidebar.multiselect("Category", sorted(df.category.unique()), sorted(df.category.unique()))

start = pd.to_datetime(start_lbl, format="%b %Y")
end = pd.to_datetime(end_lbl, format="%b %Y") + pd.offsets.MonthBegin(1)  # exclusive upper bound
d = df[
    df.region.isin(regions)
    & df.category.isin(cats)
    & (df.order_date >= start)
    & (df.order_date < end)
]

if d.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------- Header + KPIs ----------
st.title("📊 Sales Analysis & Prediction System")
st.caption(f"Showing {len(d):,} orders from {start:%b %Y} to {(end - pd.Timedelta(days=1)):%b %Y}")

total_sales, total_profit = d.sales.sum(), d.profit.sum()
orders = d.order_id.nunique()
yearly = d.groupby("year")["sales"].sum()
growth = None
if len(yearly) >= 2:
    growth = (yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2] * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Sales", f"{total_sales:,.0f}", f"{growth:+.1f}% YoY" if growth is not None else None)
c2.metric("Total Profit", f"{total_profit:,.0f}")
c3.metric("Profit Margin", f"{total_profit / total_sales * 100:.1f}%")
c4.metric("Orders", f"{orders:,}")
c5.metric("Avg Order Value", f"{total_sales / orders:,.0f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Overview", "🛍️ Products & Regions", "👥 Customers", "🔮 Predictive Studio", "🗂️ Data"]
)

# ===== Overview =====
with tab1:
    monthly_o = d.set_index("order_date").resample("MS")["sales"].sum().reset_index()
    monthly_o["3-mo avg"] = monthly_o["sales"].rolling(3).mean()
    fig = px.line(monthly_o, x="order_date", y=["sales", "3-mo avg"],
                  labels={"order_date": "Month", "value": "Sales", "variable": ""},
                  title="Monthly Sales Trend")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    yr = d.groupby("year")[["sales", "profit"]].sum().reset_index()
    fig = px.bar(yr, x="year", y=["sales", "profit"], barmode="group",
                 title="Sales & Profit by Year", labels={"value": "Amount", "variable": ""})
    fig.update_xaxes(type="category")
    left.plotly_chart(fig, width="stretch")

    pivot = d.pivot_table(values="sales", index="month", columns="year", aggfunc="sum")
    fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale="YlGnBu", aspect="auto",
                    title="Seasonality Heatmap (Month x Year)")
    right.plotly_chart(fig, width="stretch")

# ===== Products & Regions =====
with tab2:
    left, right = st.columns(2)
    prod = d.groupby("product")["sales"].sum().sort_values().reset_index()
    left.plotly_chart(px.bar(prod, x="sales", y="product", orientation="h", title="Sales by Product",
                             color="sales", color_continuous_scale="Blues"), width="stretch")
    reg = d.groupby("region")["sales"].sum().reset_index()
    right.plotly_chart(px.pie(reg, names="region", values="sales", hole=0.45, title="Sales by Region"),
                       width="stretch")

    left, right = st.columns(2)
    cat = d.groupby("category")[["sales", "profit"]].sum().reset_index()
    left.plotly_chart(px.bar(cat, x="category", y=["sales", "profit"], barmode="group",
                             title="Category: Sales vs Profit", labels={"value": "Amount", "variable": ""}),
                      width="stretch")
    right.plotly_chart(px.sunburst(d, path=["category", "product"], values="sales",
                                   title="Category > Product Breakdown"), width="stretch")

# ===== Customers =====
with tab3:
    st.subheader("Customer Segmentation (RFM + K-Means)")
    rfm = build_rfm(d, d.order_date.max() + pd.Timedelta(days=1))
    if len(rfm) < 10:
        st.warning(f"Only {len(rfm)} customer(s) match the current filters, so segments are not "
                   "meaningful. Widen the date range, region or category in the sidebar.")

    summary = rfm.groupby("Segment").agg(
        Customers=("customer_id", "count"),
        Avg_Recency_days=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum"),
    ).round(1).reset_index()

    left, right = st.columns(2)
    left.plotly_chart(px.scatter(rfm, x="Frequency", y="Monetary", color="Segment",
                                 hover_data=["customer_id", "Recency"], title="Customer Segments"),
                      width="stretch")
    right.plotly_chart(px.pie(summary, names="Segment", values="Total_Revenue", hole=0.45,
                              title="Revenue Share by Segment"), width="stretch")
    st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("🔍 Customer Lookup")
    cid = st.selectbox("Pick a customer ID", sorted(rfm.customer_id.unique()))
    row = rfm[rfm.customer_id == cid].iloc[0]
    actions = {
        "Champions": "Reward with loyalty perks and early access to new products.",
        "Regular": "Upsell and cross-sell bundles to raise order value.",
        "At-Risk": "Send a win-back offer: they haven't purchased in a long time.",
    }
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Segment", row.Segment)
    k2.metric("Days since last order", f"{row.Recency:.0f}")
    k3.metric("Orders", f"{row.Frequency:.0f}")
    k4.metric("Total spend", f"{row.Monetary:,.0f}")
    st.info(f"**Recommended action:** {actions[row.Segment]}")

# ===== Predictive Studio =====
with tab4:
    st.subheader("🔮 Predictive Studio")
    st.caption("Models are trained live on whatever filters you set in the sidebar.")

    use_period = st.checkbox("Train only on the sidebar period", value=False,
                             help="Off = the model learns from all history for the selected regions "
                                  "and categories. On = it learns only from the period chosen in the sidebar.")
    src = d if use_period else df[df.region.isin(regions) & df.category.isin(cats)]
    monthly = monthly_series(src)
    if len(monthly) < 18:
        st.warning("Need at least 18 months of data to forecast. Widen the sidebar period "
                   "or untick 'Train only on the sidebar period'.")
    else:
        # --- Controls ---
        k1, k2, k3 = st.columns(3)
        horizon = k1.slider("Forecast horizon (months)", 1, 12, 6)
        uplift = k2.slider("Growth scenario (% change)", -30, 30, 0,
                           help="Simulates a marketing push (+) or a downturn (-) on all forecast months.")
        peak_boost = k3.slider("Extra Nov-Dec boost (%)", -30, 50, 0,
                               help="Simulates a bigger or smaller festive-season effect.")

        bt = backtest(monthly)
        best = bt["RMSE"].idxmin()
        choice = st.radio("Model", ["Auto (best on backtest)"] + MODELS, horizontal=True)
        model_name = best if choice.startswith("Auto") else choice

        with st.expander("📋 Model comparison (backtest on the last 6 months)", expanded=False):
            st.dataframe(bt.round(1), width="stretch")
            st.caption(f"Best model by RMSE: **{best}**. Lower MAE / RMSE / MAPE is better.")

        # --- Forecast ---
        future_dates = pd.date_range(monthly.order_date.iloc[-1] + pd.offsets.MonthBegin(1),
                                     periods=horizon, freq="MS")
        future = pd.DataFrame({"order_date": future_dates,
                               "t": np.arange(len(monthly), len(monthly) + horizon)})
        fitted, base = fit_predict(model_name, monthly, future)
        factor = (1 + uplift / 100) * np.where(future.order_date.dt.month.isin([11, 12]),
                                               1 + peak_boost / 100, 1.0)
        adj = np.clip(base * factor, 0, None)
        sigma = float(bt.loc[model_name, "RMSE"])
        hi = adj + 1.28 * sigma
        lo = np.clip(adj - 1.28 * sigma, 0, None)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly.order_date, y=monthly.sales, name="Actual",
                                 mode="lines+markers"))
        fig.add_trace(go.Scatter(x=monthly.order_date, y=fitted, name="Model fit",
                                 mode="lines", line=dict(dash="dot", color="gray")))
        fig.add_trace(go.Scatter(x=future.order_date, y=hi, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future.order_date, y=lo, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(255,0,0,0.15)",
                                 name="~80% range", hoverinfo="skip"))
        link_x = [monthly.order_date.iloc[-1]] + list(future.order_date)
        link_y = [monthly.sales.iloc[-1]] + list(adj)
        fig.add_trace(go.Scatter(x=link_x, y=link_y, name="Forecast", mode="lines+markers",
                                 line=dict(color="red", dash="dash")))
        fig.update_layout(title=f"Sales forecast - {model_name}", xaxis_title="Month",
                          yaxis_title="Sales", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        total_fc = float(adj.sum())
        last_h = float(monthly.sales.iloc[-horizon:].sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Forecast total", f"{total_fc:,.0f}", f"{(total_fc - last_h) / last_h * 100:+.1f}% vs last {horizon} mo")
        m2.metric("Best month", f"{future.order_date[int(np.argmax(adj))]:%b %Y}")
        m3.metric("Avg per month", f"{adj.mean():,.0f}")
        m4.metric("Typical error (RMSE)", f"{sigma:,.0f}")

        # --- Target tracker ---
        st.divider()
        st.subheader("🎯 Target Tracker")
        target = st.number_input(f"Sales target for the next {horizon} months",
                                 min_value=0, value=int(round(total_fc, -3)), step=5000)
        sigma_total = max(sigma * math.sqrt(horizon), 1e-9)
        z = (target - total_fc) / sigma_total
        prob = 0.5 * (1 - math.erf(z / math.sqrt(2)))  # P(actual total >= target)
        t1, t2 = st.columns(2)
        t1.metric("Gap vs forecast", f"{total_fc - target:+,.0f}")
        t2.metric("Chance of hitting target", f"{prob * 100:.0f}%")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption("Approximation: assumes forecast errors are roughly normal and independent month to month.")

        # --- Forecast by category ---
        st.divider()
        st.subheader("🧩 Forecast by Category")
        rows = []
        for c in sorted(src.category.unique()):
            mc = monthly_series(src[src.category == c], index=monthly.order_date)
            _, pc = fit_predict(model_name, mc, future)
            rows.append(pd.DataFrame({"order_date": future.order_date,
                                      "category": c,
                                      "sales": np.clip(pc * factor, 0, None)}))
        cat_fc = pd.concat(rows)
        st.plotly_chart(px.bar(cat_fc, x="order_date", y="sales", color="category", barmode="stack",
                               labels={"order_date": "Month", "sales": "Forecast sales"}),
                        width="stretch")
        st.caption("Each category is modelled separately, so the stack total can differ slightly "
                   "from the overall forecast above.")

        out = future[["order_date"]].copy()
        out["forecast"] = adj.round(0)
        out["low"] = lo.round(0)
        out["high"] = hi.round(0)
        st.download_button("⬇️ Download this forecast (CSV)", out.to_csv(index=False),
                           "forecast_scenario.csv", "text/csv")

    # --- Order estimator (works on the full dataset) ---
    st.divider()
    st.subheader("🧮 Order Estimator")
    e1, e2, e3 = st.columns(3)
    p = e1.selectbox("Product", sorted(df["product"].unique()))
    r = e2.selectbox("Region ", ["All"] + sorted(df.region.unique()))
    q = e3.slider("Quantity", 1, 10, 3)
    sub = df[df["product"] == p]
    if r != "All":
        sub = sub[sub.region == r]
    price = sub.unit_price.median()
    margin = sub.profit / sub.sales
    est_sales = price * q
    o1, o2, o3 = st.columns(3)
    o1.metric("Unit price", f"{price:,.0f}")
    o2.metric("Expected sales", f"{est_sales:,.0f}")
    o3.metric("Expected profit", f"{est_sales * margin.median():,.0f}",
              f"range {est_sales * margin.quantile(0.1):,.0f} - {est_sales * margin.quantile(0.9):,.0f}",
              delta_color="off")

# ===== Data =====
with tab5:
    st.subheader("Filtered Data")
    st.dataframe(d, width="stretch", hide_index=True)
    st.download_button("⬇️ Download filtered data", d.to_csv(index=False), "filtered_sales.csv", "text/csv")