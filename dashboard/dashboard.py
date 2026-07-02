import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Hotel Booking Cancellation Dashboard",
    layout="wide"
)

st.title("Hotel Booking Cancellation Dashboard")

st.write(
    "This dashboard supports stakeholder decision-making by showing hotel booking "
    "cancellation patterns using the cleaned dataset generated through the Agile "
    "Data Science pipeline."
)

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "processed_data" / "hotel_bookings_cleaned_q2.csv"

@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df = load_data()

st.sidebar.header("Filter Options")

hotel_options = ["All"] + sorted(df["hotel"].dropna().unique().tolist())
selected_hotel = st.sidebar.selectbox("Select Hotel Type", hotel_options)

market_options = ["All"] + sorted(df["market_segment"].dropna().unique().tolist())
selected_market = st.sidebar.selectbox("Select Market Segment", market_options)

customer_options = ["All"] + sorted(df["customer_type"].dropna().unique().tolist())
selected_customer = st.sidebar.selectbox("Select Customer Type", customer_options)

min_lead = int(df["lead_time"].min())
max_lead = int(df["lead_time"].max())

lead_time_range = st.sidebar.slider(
    "Select Lead Time Range",
    min_value=min_lead,
    max_value=max_lead,
    value=(min_lead, max_lead)
)

filtered_df = df.copy()

if selected_hotel != "All":
    filtered_df = filtered_df[filtered_df["hotel"] == selected_hotel]

if selected_market != "All":
    filtered_df = filtered_df[filtered_df["market_segment"] == selected_market]

if selected_customer != "All":
    filtered_df = filtered_df[filtered_df["customer_type"] == selected_customer]

filtered_df = filtered_df[
    (filtered_df["lead_time"] >= lead_time_range[0]) &
    (filtered_df["lead_time"] <= lead_time_range[1])
]

st.subheader("Dashboard Summary")

col1, col2, col3, col4 = st.columns(4)

total_bookings = len(filtered_df)
cancelled_bookings = int(filtered_df["is_canceled"].sum())
not_cancelled_bookings = total_bookings - cancelled_bookings
cancellation_rate = (
    cancelled_bookings / total_bookings * 100
    if total_bookings > 0 else 0
)

col1.metric("Total Bookings", f"{total_bookings:,}")
col2.metric("Cancelled Bookings", f"{cancelled_bookings:,}")
col3.metric("Not Cancelled", f"{not_cancelled_bookings:,}")
col4.metric("Cancellation Rate", f"{cancellation_rate:.2f}%")

st.divider()

st.subheader("Visualization 1: Cancellation Status Distribution")

if not filtered_df.empty:
    cancellation_distribution = (
        filtered_df["is_canceled"]
        .value_counts()
        .rename(index={0: "Not Cancelled", 1: "Cancelled"})
    )
    st.bar_chart(cancellation_distribution)
else:
    st.warning("No data available for the selected filters.")

st.subheader("Visualization 2: Cancellation Rate by Hotel Type")

if not filtered_df.empty:
    hotel_cancel_rate = (
        filtered_df.groupby("hotel")["is_canceled"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"is_canceled": "Cancellation Rate (%)"})
    )
    st.bar_chart(
        hotel_cancel_rate.set_index("hotel")
    )
else:
    st.warning("No hotel cancellation chart available.")

st.subheader("Visualization 3: Cancellation Rate by Market Segment")

if not filtered_df.empty:
    market_cancel_rate = (
        filtered_df.groupby("market_segment")["is_canceled"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"is_canceled": "Cancellation Rate (%)"})
    )
    st.bar_chart(
        market_cancel_rate.set_index("market_segment")
    )
else:
    st.warning("No market segment chart available.")

st.divider()

st.subheader("Analytical Output: Cancellation Risk Level")

if cancellation_rate >= 50:
    risk_level = "High Risk"
    recommendation = (
        "Cancellation rate is high. Stakeholders should review deposit policy, "
        "booking channels, and confirmation procedures."
    )
elif cancellation_rate >= 30:
    risk_level = "Medium Risk"
    recommendation = (
        "Cancellation rate is moderate. Stakeholders should monitor this segment "
        "and apply targeted follow-up reminders."
    )
else:
    risk_level = "Low Risk"
    recommendation = (
        "Cancellation rate is relatively low. Current booking segment appears stable."
    )

st.metric("Selected Segment Risk Level", risk_level)
st.write("Recommendation:", recommendation)

st.subheader("Filtered Booking Records")
st.dataframe(filtered_df)
