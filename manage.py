import pandas as pd
from pathlib import Path
from column_cleaner import EarthquakeDataCleaner
from data_audit import audit_summary
from visualizer import *
from functools import partial

# Load the dataset
LOCAL_FILE = Path("Files/query.csv")
raw = pd.read_csv(LOCAL_FILE)

# Run the cleaning pipeline
cleaner = EarthquakeDataCleaner(raw)
copy = cleaner.run_pipeline()


# Column summary
audit_report = audit_summary(copy)

# Quick unique value review for low-cardinality columns (< 10 unique values)
for i in copy.columns:
    if copy[i].nunique(dropna=False) < 10:
        uniq_vals = i
        uniq_vals_num = copy[i].nunique(dropna=False)
        uniq_vals_list = copy[i].unique().tolist()
        uniq_list = (uniq_vals, uniq_vals_num, uniq_vals_list)






# visualizer
plot_mag = plot_earthquake_magnitudes
country_mags = mag_by_country



print(copy.head(10).to_string())




# Working with columns safe test configuration reference
safe_test_columns = ['magType', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']




if __name__ == "__main__":
    pass