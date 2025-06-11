import sys
import pandas as pd
from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset
import os

# Load the reference and current data
current_path = "data/preprocessed/price/AAPL.csv"
reference_path = "data/reference/price/AAPL.csv"

current = pd.read_csv(current_path)

if not os.path.exists(reference_path):
    print(f"Reference file not found. Copying from current data to {reference_path}.")
    os.makedirs(os.path.dirname(reference_path), exist_ok=True)
    current.to_csv(reference_path, index=False)

reference = pd.read_csv(reference_path)

if "datetime" in reference.columns:
    del reference["datetime"] # We won't test datetime so we delete it
if "datetime" in current.columns:
    del current["datetime"]

# Check if the reference and current data have the same columns
report = Report([
        DataSummaryPreset(), # Calculates descriptive statistics of data
        DataDriftPreset(),
    ],
    include_tests=True
)

# Run the report on the reference and current data
result = report.run(reference_data=reference, current_data=current)

# Save the report to a HTML file
html_filepath = "reports/price/data_testing_report.html"
os.makedirs(os.path.dirname(html_filepath), exist_ok=True)
result.save_html(html_filepath)

# Check if the report contains any tests and if all tests passed
all_tests_passed = True
result_dict = result.dict()
if "tests" in result_dict:
    for test in result_dict["tests"]:
        if "status" in test and test["status"] != "SUCCESS":
            all_tests_passed = False
            break

if not all_tests_passed:
    print("Data tests failed.")
    sys.exit(1)
else:
    print("Data tests passed.")
    # Replace the reference data with the current data
    os.remove(reference_path)
    current = pd.read_csv(current_path)
    current.to_csv(reference_path, index=False)
    sys.exit(0)
