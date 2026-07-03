import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def find_cleaned_dataset():
    possible_paths = [
        "processed_data/hotel_bookings_cleaned_q2.csv",
        "processed_data/hotel_bookings_cleaned.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Cleaned dataset not found. Please check the processed_data folder."
    )


def train_cancellation_model():
    data_path = find_cleaned_dataset()
    df = pd.read_csv(data_path)

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

    missing_features = [
        col for col in selected_features if col not in df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

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
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
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

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

    metadata = {
        "features": selected_features,
        "hotel_options": sorted(df["hotel"].dropna().unique().tolist()),
        "market_segment_options": sorted(df["market_segment"].dropna().unique().tolist()),
        "deposit_type_options": sorted(df["deposit_type"].dropna().unique().tolist()),
        "customer_type_options": sorted(df["customer_type"].dropna().unique().tolist()),
        "lead_time_max": int(df["lead_time"].max()),
        "adr_max": float(df["adr"].max()),
        "metrics": metrics,
        "training_data_path": data_path
    }

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/cancellation_model.pkl")
    joblib.dump(metadata, "models/model_metadata.pkl")

    print("Cancellation prediction model trained successfully.")
    print("Training dataset:", data_path)
    print("Model saved to: models/cancellation_model.pkl")
    print("Metadata saved to: models/model_metadata.pkl")
    print("\nModel Evaluation Metrics")
    print("=" * 50)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    train_cancellation_model()
