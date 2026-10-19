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
mag_dist = plot_magnitude_distribution
depth_dist = plot_depth_distribution
corr_hm = plot_corr_heatmap
mag_cnt = plot_mag_by_country
event_by_year = plot_events_per_year
mag_by_decade = plot_mean_magnitude_decade_heatmap


# print(copy.head(1).to_string())




# Working with columns safe test configuration reference
safe_test_columns = ['magType', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']







if __name__ == "__main__":
    pass