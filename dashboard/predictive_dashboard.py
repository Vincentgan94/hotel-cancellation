import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictive Dashboard",
    layout="wide"
)

st.title("Hotel Booking Cancellation Predictive Dashboard")

st.write(
    "This dashboard supports stakeholder decision-making by combining "
    "cancellation visualizations, interactive filters, and predictive "
    "cancellation-risk output."
)

BASE_DIR = Path(__file__).resolve().parent.parent

cleaned_data_paths = [
    BASE_DIR / "processed_data" / "hotel_bookings_cleaned_q2.csv",
    BASE_DIR / "processed_data" / "hotel_bookings_cleaned.csv"
]

data_path = None
for path in cleaned_data_paths:
    if path.exists():
        data_path = path
        break

if data_path is None:
    st.error("Cleaned dataset not found in processed_data folder.")
    st.stop()


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def train_model(df):
    selected_features = [
        "hotel",
        "lead_time",
        "market_segment",
        "deposit_type",
        "customer_type",
        "total_guests",
        "total_stay_nights",
        "adr",
        "previous_cancellations",
        "booking_changes",
        "required_car_parking_spaces",
        "total_of_special_requests"
    ]

    target = "is_canceled"

    X = df[selected_features]
    y = df[target]

    categorical_features = [
        "hotel",
        "market_segment",
        "deposit_type",
        "customer_type"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    metadata = {
        "features": selected_features,
        "hotel_options": sorted(df["hotel"].dropna().unique().tolist()),
        "market_segment_options": sorted(df["market_segment"].dropna().unique().tolist()),
        "deposit_type_options": sorted(df["deposit_type"].dropna().unique().tolist()),
        "customer_type_options": sorted(df["customer_type"].dropna().unique().tolist()),
        "lead_time_max": int(df["lead_time"].max()),
        "adr_max": float(df["adr"].max())
    }

    return model, metadata


df = load_data(data_path)
model, metadata = train_model(df)

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
cancelled_bookings = int(filtered_df["is_canceled"].sum()) if total_bookings > 0 else 0
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

    distribution_table = cancellation_distribution.reset_index()
    distribution_table.columns = ["Cancellation Status", "Count"]
    st.dataframe(distribution_table)
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

    st.bar_chart(hotel_cancel_rate.set_index("hotel"))
    st.dataframe(hotel_cancel_rate)
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

    st.bar_chart(market_cancel_rate.set_index("market_segment"))
    st.dataframe(market_cancel_rate)
else:
    st.warning("No market segment chart available.")

st.divider()

st.subheader("Analytical Output: Selected Segment Risk Level")

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

st.divider()

st.subheader("Predictive Output: Booking Cancellation Prediction")

with st.form("prediction_form"):
    col_a, col_b = st.columns(2)

    with col_a:
        pred_hotel = st.selectbox("Hotel Type", metadata["hotel_options"])
        pred_market_segment = st.selectbox("Market Segment", metadata["market_segment_options"])
        pred_deposit_type = st.selectbox("Deposit Type", metadata["deposit_type_options"])
        pred_customer_type = st.selectbox("Customer Type", metadata["customer_type_options"])

        pred_lead_time = st.number_input(
            "Lead Time",
            min_value=0,
            max_value=int(metadata["lead_time_max"]),
            value=30
        )

        pred_adr = st.number_input(
            "Average Daily Rate (ADR)",
            min_value=0.0,
            max_value=float(metadata["adr_max"]),
            value=100.0
        )

    with col_b:
        pred_total_guests = st.number_input("Total Guests", min_value=1, max_value=10, value=2)
        pred_total_stay_nights = st.number_input("Total Stay Nights", min_value=1, max_value=30, value=2)
        pred_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=30, value=0)
        pred_booking_changes = st.number_input("Booking Changes", min_value=0, max_value=30, value=0)
        pred_parking = st.number_input("Required Car Parking Spaces", min_value=0, max_value=10, value=0)
        pred_special_requests = st.number_input("Total Special Requests", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("Predict Cancellation Risk")

if submitted:
    input_df = pd.DataFrame({
        "hotel": [pred_hotel],
        "lead_time": [pred_lead_time],
        "market_segment": [pred_market_segment],
        "deposit_type": [pred_deposit_type],
        "customer_type": [pred_customer_type],
        "total_guests": [pred_total_guests],
        "total_stay_nights": [pred_total_stay_nights],
        "adr": [pred_adr],
        "previous_cancellations": [pred_previous_cancellations],
        "booking_changes": [pred_booking_changes],
        "required_car_parking_spaces": [pred_parking],
        "total_of_special_requests": [pred_special_requests]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    predicted_label = "Cancelled" if prediction == 1 else "Not Cancelled"

    if probability >= 0.50:
        prediction_risk = "High Risk"
    elif probability >= 0.30:
        prediction_risk = "Medium Risk"
    else:
        prediction_risk = "Low Risk"

    st.metric("Predicted Cancellation Outcome", predicted_label)
    st.metric("Predicted Cancellation Probability", f"{probability:.2%}")
    st.metric("Prediction Risk Category", prediction_risk)

    if predicted_label == "Cancelled":
        st.warning(
            "This booking scenario is predicted as likely to be cancelled. "
            "Stakeholders may review booking confirmation, deposit policy, or follow-up action."
        )
    else:
        st.success(
            "This booking scenario is predicted as not cancelled. "
            "The booking appears relatively stable based on the model output."
        )

st.divider()

st.subheader("Filtered Booking Records")
st.dataframe(filtered_df)
