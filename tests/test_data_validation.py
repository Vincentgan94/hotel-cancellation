import os
import sys
import pandas as pd

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)

from data_validation import validate_hotel_booking_data
from data_validation import clean_hotel_booking_data_q2


def test_validation_script_runs_successfully():
    file_path = "data/hotel_bookings.csv"

    assert os.path.exists(file_path), "Dataset file does not exist."

    results = validate_hotel_booking_data(file_path)

    assert isinstance(results, pd.DataFrame)
    assert results.shape[0] == 10
    assert "Validation Check" in results.columns
    assert "Result" in results.columns
    assert "Status" in results.columns


def test_raw_dataset_loaded_correctly():
    results = validate_hotel_booking_data("data/hotel_bookings.csv")

    row = results[results["Validation Check"] == "Dataset loaded"].iloc[0]

    assert row["Status"] == "PASS"
    assert row["Result"] == "119390 rows and 32 columns"


def test_required_columns_and_target_validity():
    results = validate_hotel_booking_data("data/hotel_bookings.csv")

    required_columns_row = results[
        results["Validation Check"] == "Required columns exist"
    ].iloc[0]

    target_row = results[
        results["Validation Check"] == "Target variable validity"
    ].iloc[0]

    assert required_columns_row["Status"] == "PASS"
    assert required_columns_row["Result"] == "All required columns are available"

    assert target_row["Status"] == "PASS"
    assert target_row["Result"] == "Unique values in is_canceled: [0, 1]"


def test_known_data_quality_warnings():
    results = validate_hotel_booking_data("data/hotel_bookings.csv")

    duplicate_row = results[
        results["Validation Check"] == "Duplicate records"
    ].iloc[0]

    guest_row = results[
        results["Validation Check"] == "Invalid guest count"
    ].iloc[0]

    zero_stay_row = results[
        results["Validation Check"] == "Zero total stay nights"
    ].iloc[0]

    adr_row = results[
        results["Validation Check"] == "ADR value range"
    ].iloc[0]

    assert duplicate_row["Status"] == "WARNING"
    assert duplicate_row["Result"] == "31994 duplicate rows found"

    assert guest_row["Status"] == "WARNING"
    assert guest_row["Result"] == "180 rows with zero total guests"

    assert zero_stay_row["Status"] == "WARNING"
    assert zero_stay_row["Result"] == "715 rows with zero total stay nights"

    assert adr_row["Status"] == "WARNING"
    assert adr_row["Result"] == "1 negative ADR rows; 1 ADR rows above 1000"


def test_date_parsing_and_leakage_check():
    results = validate_hotel_booking_data("data/hotel_bookings.csv")

    date_row = results[
        results["Validation Check"] == "Reservation status date parsing"
    ].iloc[0]

    leakage_row = results[
        results["Validation Check"] == "Target leakage column check"
    ].iloc[0]

    assert date_row["Status"] == "PASS"
    assert date_row["Result"] == "0 invalid parsed dates"

    assert leakage_row["Status"] == "WARNING"
    assert leakage_row["Result"] == (
        "Leakage-prone columns found: "
        "['reservation_status', 'reservation_status_date']"
    )


def test_q2_cleaning_output_matches_expected_result():
    cleaned_df, summary = clean_hotel_booking_data_q2(
        "data/hotel_bookings.csv",
        "processed_data/hotel_bookings_cleaned_q2.csv"
    )

    assert cleaned_df.shape == (86637, 34)
    assert summary["Original Rows"] == 119390
    assert summary["Original Columns"] == 32
    assert summary["Duplicate Rows Removed"] == 31994
    assert summary["Cleaned Rows"] == 86637
    assert summary["Cleaned Columns"] == 34
    assert summary["Not Cancelled Count"] == 62652
    assert summary["Cancelled Count"] == 23985

    assert cleaned_df.duplicated().sum() == 0
    assert (cleaned_df["total_guests"] <= 0).sum() == 0
    assert (cleaned_df["total_stay_nights"] <= 0).sum() == 0
    assert ((cleaned_df["adr"] < 0) | (cleaned_df["adr"] > 1000)).sum() == 0
