import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Customer Retention Dashboard",
    layout="wide"
)

st.title("📊 Customer Retention & Intelligence Dashboard")
st.caption("Executive View – Fashion E-Commerce Startup")

# ======================================
# LOAD & CLEAN DATA
# ======================================
@st.cache_data
def load_data():
    df = pd.read_excel("Data.xlsx")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    def clean_binary(col):
        return (
            col.astype(str)
            .str.strip()
            .str.lower()
            .map({'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0})
        )

    df['Return_Visit_Num'] = clean_binary(df['Return_Visit'])
    df['Discount_Used_Num'] = clean_binary(df['Discount_Used'])
    df['Email_Engagement_Num'] = clean_binary(df['Email_Engagement'])

    df['Season'] = (df['Date'].dt.month % 12 // 3 + 1).map({
        1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'
    })

    return df

df = load_data()

# ======================================
# SIDEBAR — CLICK-ONLY FILTERS
# ======================================
st.sidebar.markdown("## 🔎 Filters")
st.sidebar.caption("Click to adjust data scope")

# ---- DATE SLIDER ----
min_date = df['Date'].min()
max_date = df['Date'].max()

date_range = st.sidebar.slider(
    "📅 Date Range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM-DD"
)

st.sidebar.markdown("---")

# ---- SEASON FILTER (CHECKBOXES) ----
st.sidebar.markdown("### 🗓️ Season")

all_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
season_filter = []

for season in all_seasons:
    if st.sidebar.checkbox(season, value=True):
        season_filter.append(season)

if not season_filter:
    season_filter = all_seasons

st.sidebar.markdown("---")

# ---- CATEGORY FILTER (CHECKBOXES) ----
st.sidebar.markdown("### 👗 Category")

all_categories = sorted(df['Category'].dropna().unique())
category_filter = []

for cat in all_categories:
    if st.sidebar.checkbox(cat, value=True):
        category_filter.append(cat)

if not category_filter:
    category_filter = all_categories

# ---- APPLY FILTERS ----
df_filtered = df[
    (df['Date'] >= date_range[0]) &
    (df['Date'] <= date_range[1]) &
    (df['Season'].isin(season_filter)) &
    (df['Category'].isin(category_filter))
]

# ======================================
# KPI ROW
# ======================================
k1, k2, k3, k4 = st.columns(4)

k1.metric("Retention Rate", f"{df_filtered['Return_Visit_Num'].mean()*100:.1f}%")
k2.metric("Total Revenue", f"${df_filtered['Purchase_Value'].sum():,.0f}")
k3.metric("Avg Order Value", f"${df_filtered['Purchase_Value'].mean():.0f}")
k4.metric("Customers", df_filtered['Customer_ID'].nunique())

st.caption(
    f"Showing **{len(df_filtered):,} transactions** "
    f"out of **{len(df):,} total**"
)

st.markdown("---")

# ======================================
# REVENUE SPLIT
# ======================================
st.subheader("💰 Revenue: New vs Returning Customers")

rev_split = (
    df_filtered.groupby('Return_Visit_Num')['Purchase_Value']
    .sum()
    .reset_index()
)

rev_split['Customer Type'] = rev_split['Return_Visit_Num'].map({
    0: 'One-time',
    1: 'Returning'
})

fig = px.pie(
    rev_split,
    names='Customer Type',
    values='Purchase_Value',
    hole=0.45,
    title="Revenue Contribution"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "Returning customers generate a disproportionately large share of revenue, "
    "making retention more cost-effective than acquisition."
)

# ======================================
# PROMOTION ROI
# ======================================
st.subheader("🎯 Promotion ROI")

promo_roi = (
    df_filtered.groupby('Discount_Used_Num')
    .agg(
        Avg_Order_Value=('Purchase_Value', 'mean'),
        Retention_Rate=('Return_Visit_Num', 'mean')
    )
    .reset_index()
)

promo_roi['Discount'] = promo_roi['Discount_Used_Num'].map({
    0: 'No Discount',
    1: 'Discount Used'
})

fig = px.bar(
    promo_roi,
    x='Discount',
    y=['Avg_Order_Value', 'Retention_Rate'],
    barmode='group',
    title="Impact of Discounts"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "Discounts increase order value and retention, but ROI is strongest "
    "when promotions are personalized."
)

# ======================================
# SEASONAL BUYING TRENDS
# ======================================
st.subheader("🗓️ Seasonal Buying Trends")

seasonal_revenue = (
    df_filtered.groupby('Season')['Purchase_Value']
    .sum()
    .reindex(['Winter', 'Spring', 'Summer', 'Fall'])
    .reset_index()
)

fig = px.bar(
    seasonal_revenue,
    x='Season',
    y='Purchase_Value',
    title="Seasonal Revenue Trend"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "Customer spending follows clear seasonal patterns. "
    "Peak seasons are ideal for inventory expansion and campaign launches."
)

# ======================================
# CATEGORY PERFORMANCE
# ======================================
st.subheader("👗 Revenue by Product Category")

category_revenue = (
    df_filtered.groupby('Category')['Purchase_Value']
    .sum()
    .reset_index()
    .sort_values(by='Purchase_Value', ascending=False)
)

fig = px.bar(
    category_revenue,
    x='Category',
    y='Purchase_Value',
    title="Category Revenue Performance"
)

st.plotly_chart(fig, use_container_width=True)

# ======================================
# CHURN RISK MODEL
# ======================================
st.subheader("⚠️ Churn Risk Overview")

last_purchase = df.groupby('Customer_ID')['Date'].max()
days_since_last = (df['Date'].max() - last_purchase).dt.days

churn_df = df.groupby('Customer_ID').agg(
    Avg_Purchase=('Purchase_Value', 'mean'),
    Orders=('Date', 'count'),
    Email_Engagement=('Email_Engagement_Num', 'mean'),
    Discount_Usage=('Discount_Used_Num', 'mean'),
    Returned=('Return_Visit_Num', 'min')
)

churn_df['Days_Since_Last_Purchase'] = days_since_last
churn_df['Churn'] = (churn_df['Returned'] == 0).astype(int)

X = churn_df.drop(['Churn', 'Returned'], axis=1)
y = churn_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

churn_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', GradientBoostingClassifier())
])

churn_pipeline.fit(X_train, y_train)
churn_df['Churn_Probability'] = churn_pipeline.predict_proba(X)[:, 1]

roc = roc_auc_score(y_test, churn_pipeline.predict_proba(X_test)[:, 1])
st.metric("Churn Model ROC-AUC", f"{roc:.2f}")

# ======================================
# CLV MODEL
# ======================================
st.subheader("🔮 Customer Lifetime Value (CLV)")

clv_df = df.groupby('Customer_ID').agg(
    Frequency=('Date', 'count'),
    Monetary=('Purchase_Value', 'mean'),
    Email_Engagement=('Email_Engagement_Num', 'mean'),
    Discount_Usage=('Discount_Used_Num', 'mean')
)

clv_df['CLV'] = clv_df['Frequency'] * clv_df['Monetary']

X = clv_df.drop('CLV', axis=1)
y = clv_df['CLV']

clv_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42))
])

clv_pipeline.fit(X, y)
clv_df['Predicted_CLV'] = clv_pipeline.predict(X)

# ======================================
# PRIORITY CUSTOMERS
# ======================================
st.subheader("🔥 High-Priority Customers")

priority_df = churn_df[['Churn_Probability']].join(
    clv_df[['Predicted_CLV']]
)

priority_df['Priority_Score'] = (
    (priority_df['Predicted_CLV'] / priority_df['Predicted_CLV'].max()) *
    priority_df['Churn_Probability']
)

st.dataframe(
    priority_df.sort_values('Priority_Score', ascending=False).head(10),
    use_container_width=True
)

st.success(
    "These customers combine **high future value** with **high churn risk** "
    "and should be targeted first with personalized retention actions."
)
