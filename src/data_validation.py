import os
import pandas as pd


def validate_hotel_booking_data(file_path):
    """
    Automated validation script for the hotel booking cancellation dataset.
    This script checks important data-quality issues before feature engineering,
    modelling, and dashboard development.
    """

    df = pd.read_csv(file_path)

    validation_results = []

    # 1. Dataset loading check
    validation_results.append({
        "Validation Check": "Dataset loaded",
        "Result": f"{df.shape[0]} rows and {df.shape[1]} columns",
        "Status": "PASS"
    })

    # 2. Required column check
    required_columns = [
        "hotel",
        "is_canceled",
        "lead_time",
        "adults",
        "children",
        "babies",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "country",
        "market_segment",
        "distribution_channel",
        "deposit_type",
        "customer_type",
        "adr",
        "reservation_status",
        "reservation_status_date"
    ]

    missing_required_columns = [
        col for col in required_columns if col not in df.columns
    ]

    validation_results.append({
        "Validation Check": "Required columns exist",
        "Result": "All required columns are available"
        if len(missing_required_columns) == 0
        else str(missing_required_columns),
        "Status": "PASS" if len(missing_required_columns) == 0 else "FAIL"
    })

    # 3. Target variable validity check
    target_values = sorted(df["is_canceled"].dropna().unique().tolist())

    validation_results.append({
        "Validation Check": "Target variable validity",
        "Result": f"Unique values in is_canceled: {target_values}",
        "Status": "PASS" if target_values == [0, 1] else "FAIL"
    })

    # 4. Missing-value check
    missing_values = df.isna().sum()
    missing_columns = missing_values[missing_values > 0]

    validation_results.append({
        "Validation Check": "Missing values",
        "Result": missing_columns.to_dict()
        if len(missing_columns) > 0
        else "No missing values found",
        "Status": "WARNING" if len(missing_columns) > 0 else "PASS"
    })

    # 5. Duplicate row check
    duplicate_count = df.duplicated().sum()

    validation_results.append({
        "Validation Check": "Duplicate records",
        "Result": f"{duplicate_count} duplicate rows found",
        "Status": "WARNING" if duplicate_count > 0 else "PASS"
    })

    # 6. Invalid guest-count check
    invalid_guest_count = (
        (df["adults"] + df["children"].fillna(0) + df["babies"]) == 0
    ).sum()

    validation_results.append({
        "Validation Check": "Invalid guest count",
        "Result": f"{invalid_guest_count} rows with zero total guests",
        "Status": "WARNING" if invalid_guest_count > 0 else "PASS"
    })

    # 7. Zero total stay-night check
    total_stay_nights = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    )

    zero_stay_count = (total_stay_nights == 0).sum()

    validation_results.append({
        "Validation Check": "Zero total stay nights",
        "Result": f"{zero_stay_count} rows with zero total stay nights",
        "Status": "WARNING" if zero_stay_count > 0 else "PASS"
    })

    # 8. ADR value range check
    negative_adr_count = (df["adr"] < 0).sum()
    extreme_adr_count = (df["adr"] > 1000).sum()

    validation_results.append({
        "Validation Check": "ADR value range",
        "Result": (
            f"{negative_adr_count} negative ADR rows; "
            f"{extreme_adr_count} ADR rows above 1000"
        ),
        "Status": "WARNING"
        if negative_adr_count > 0 or extreme_adr_count > 0
        else "PASS"
    })

    # 9. Date parsing check
    parsed_dates = pd.to_datetime(
        df["reservation_status_date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    invalid_date_count = parsed_dates.isna().sum()

    validation_results.append({
        "Validation Check": "Reservation status date parsing",
        "Result": f"{invalid_date_count} invalid parsed dates",
        "Status": "PASS" if invalid_date_count == 0 else "WARNING"
    })

    # 10. Target leakage check
    leakage_columns = ["reservation_status", "reservation_status_date"]
    existing_leakage_columns = [
        col for col in leakage_columns if col in df.columns
    ]

    validation_results.append({
        "Validation Check": "Target leakage column check",
        "Result": f"Leakage-prone columns found: {existing_leakage_columns}",
        "Status": "WARNING" if len(existing_leakage_columns) > 0 else "PASS"
    })

    return pd.DataFrame(validation_results)


def clean_hotel_booking_data_q2(
    file_path,
    output_path="processed_data/hotel_bookings_cleaned_q2.csv"
):
    """
    Applies the same cleaning process used in Q2.
    The expected cleaned dataset shape is 86637 rows and 34 columns.
    """

    df = pd.read_csv(file_path)

    original_rows = df.shape[0]
    original_columns = df.shape[1]

    df_clean = df.copy()

    # Same as Q2: handle children missing values before creating total_guests
    df_clean["children"] = df_clean["children"].fillna(0)

    # Same as Q2: create derived columns
    df_clean["total_guests"] = (
        df_clean["adults"] + df_clean["children"] + df_clean["babies"]
    )

    df_clean["total_stay_nights"] = (
        df_clean["stays_in_weekend_nights"] + df_clean["stays_in_week_nights"]
    )

    duplicate_rows = df_clean.duplicated().sum()

    # Same as Q2: remove duplicate rows
    df_clean = df_clean.drop_duplicates()

    # Same as Q2: remove invalid records
    df_clean = df_clean[
        (df_clean["total_guests"] > 0) &
        (df_clean["total_stay_nights"] > 0) &
        (df_clean["adr"] >= 0) &
        (df_clean["adr"] <= 1000)
    ].copy()

    # Same as Q2: handle country missing values
    df_clean["country"] = df_clean["country"].fillna("Unknown")

    # Same as Q2: standardize undefined categories
    for col in ["meal", "market_segment", "distribution_channel"]:
        df_clean[col] = df_clean[col].replace("Undefined", "Unknown")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    target_counts = df_clean["is_canceled"].value_counts().to_dict()

    summary = {
        "Original Rows": original_rows,
        "Original Columns": original_columns,
        "Duplicate Rows Removed": int(duplicate_rows),
        "Cleaned Rows": df_clean.shape[0],
        "Cleaned Columns": df_clean.shape[1],
        "Not Cancelled Count": int(target_counts.get(0, 0)),
        "Cancelled Count": int(target_counts.get(1, 0)),
        "Output File": output_path
    }

    return df_clean, summary


if __name__ == "__main__":
    file_path = "data/hotel_bookings.csv"

    os.makedirs("validation_reports", exist_ok=True)
    os.makedirs("processed_data", exist_ok=True)

    validation_results = validate_hotel_booking_data(file_path)
    validation_results.to_csv(
        "validation_reports/validation_report.csv",
        index=False
    )

    cleaned_df, cleaning_summary = clean_hotel_booking_data_q2(
        file_path,
        "processed_data/hotel_bookings_cleaned_q2.csv"
    )

    cleaning_summary_df = pd.DataFrame([cleaning_summary])
    cleaning_summary_df.to_csv(
        "validation_reports/q2_cleaning_summary.csv",
        index=False
    )

    print("\nAutomated Data Validation Results")
    print("=" * 80)
    print(validation_results.to_string(index=False))

    print("\nQ2 Cleaning Summary")
    print("=" * 80)
    print(cleaning_summary_df.to_string(index=False))
