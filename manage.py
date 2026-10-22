import pandas as pd
from pathlib import Path
from column_cleaner import EarthquakeDataCleaner
from data_audit import audit_summary
from visualizer import *
from functools import partial
from scipy import stats




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




# ======visualizer======

# Category 1: Basic Data Summary & Exploratory Analysis
magnitude_distribution = plot_magnitude_distribution
depth_distribution = plot_depth_distribution
events_per_year = plot_events_per_year
desc_table = plot_desc_table
missing_percent = plot_missing_percent
depth_hist = plot_depth_hist # give raw to depth_hist

# Category 2: Bivariate & Spatial Explorations
mag_by_country = plot_mag_by_country
earthquake_map = plot_earthquake_map
pearson_spearman = plot_pearson_spearman

# Category 3: Advanced Multivariate & Temporal Analysis
corr_heatmap = plot_corr_heatmap
mean_magnitude_decade_heatmap = plot_mean_magnitude_decade_heatmap
quarterly_series = plot_quarterly_series


# Category 4: Statistical Hypothesis Testing & Seismological Modeling
shapiro_test = plot_shapiro_test # give copy and col name
gutenberg_richter = plot_gutenberg_richter
poisson = plot_poisson
poisson_by_country = plot_poisson_by_country
era_mannwhitney = plot_era_mannwhitney
country_kruskal = plot_country_kruskal


# print(copy.head(1).to_string())




# Working with columns safe test configuration reference
safe_test_columns = ['magType', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']







if __name__ == "__main__":
    pass
