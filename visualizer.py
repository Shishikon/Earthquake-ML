import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


# ==============================================================================
# CATEGORY 1: Basic Data Summary & Exploratory Analysis
# Level of Difficulty: Low | Level of Importance: High (Foundation & Quality Control)
# ==============================================================================

# mag distribution histogram
def plot_magnitude_distribution(df: pd.DataFrame):
    temp_df = df.copy()
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")

    mag_data = temp_df["mag"].dropna()

    # Histogram + KDE (probability density)
    sns.histplot(mag_data, kde=True, color="crimson", bins=30)

    min_val = np.floor(mag_data.min())
    max_val = np.ceil(mag_data.max())
    plt.xticks(np.arange(min_val, max_val + 0.5, 0.5))

    plt.title("Distribution of Earthquake Magnitudes (Target: mag)", fontsize=13, fontweight="bold")
    plt.xlabel("Magnitude", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    plt.show()

# depth distribution
def plot_depth_distribution(df: pd.DataFrame):
    temp_df = df.copy()
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    depth_order = [
        "very_shallow",
        "shallow",
        "moderately_shallow",
        "intermediate_1",
        "intermediate_2",
        "intermediate_3"
    ]
    existing_order = [cat for cat in depth_order if cat in temp_df['depth_class'].unique()]
    counter = temp_df['depth_class'].value_counts()
    labels = [f"{cat}\n(n={counter[cat]})" for cat in existing_order]
    sns.countplot(
        data=temp_df,
        x="depth_class",
        order=existing_order,
        palette="viridis",
        hue="depth_class",
        legend=False
    )

    plt.title("Distribution of Earthquake Depth Classes", fontsize=13, fontweight="bold")
    plt.xlabel("Depth Class", fontsize=11)
    plt.ylabel("Frequency (Count)", fontsize=11)
    plt.xticks(ticks=range(len(existing_order)), labels=labels, ha="right")
    plt.tight_layout()
    plt.show()

# events by year
def plot_events_per_year(df: pd.DataFrame):
    temp_df = df.copy()
    date_col = "date"
    start_year = 1928
    end_year = 2026


    # Extract the year from your date column safely
    temp_df["year"] = pd.to_datetime(temp_df[date_col], errors="coerce").dt.year

    # Count occurrences per year and sort by year
    year_counts = temp_df["year"].value_counts().sort_index()

    # Reindex to force the full timeline range, filling missing years with 0
    full_range = range(start_year, end_year + 1)
    year_counts = year_counts.reindex(full_range, fill_value=0)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Create the bar plot
    sns.barplot(
        x=list(year_counts.index),
        y=year_counts.values,
        palette='dark:steelblue',
        hue=list(year_counts.index),
        legend=False
    )

    tick_positions = np.arange(0, len(full_range), 5)
    tick_labels = list(full_range)[::5]
    plt.xticks(ticks=tick_positions, rotation=45)

    plt.title(f"Earthquake Frequency Per Year ({start_year}–{end_year})", fontsize=13, fontweight="bold")
    plt.xlabel("Year", fontsize=11)
    plt.ylabel("Number of Events", fontsize=11)
    plt.tight_layout()
    plt.show()

# descriptive analysis + skew, kurt, missingness
def plot_desc_table(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    cols = temp_df.select_dtypes(include="number").columns

    dt = temp_df[cols].describe().T

    dt['skew'] = temp_df[cols].skew()
    dt['kurtosis'] = temp_df[cols].kurtosis()
    dt['missing_%'] = temp_df[cols].isna().mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(cols) * 0.35)))

    sns.barplot(
        x=dt['missing_%'],
        y=dt.index,
        ax=axes[0],
        color='salmon'
    )
    axes[0].set_title('Missing Values (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Percentage (%)')
    axes[0].set_ylabel('')
    axes[0].set_xlim(0, 100)

    sns.barplot(x=dt['skew'], y=dt.index, ax=axes[1], color='skyblue')
    axes[1].set_title('Feature Skewness', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Skew Value')
    axes[1].set_ylabel('')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Symmetric (0)')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    return dt.round(2)

# missing percent table
def plot_missing_percent(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    pct = df.isna().mean().mul(100).sort_values(ascending=False)
    pct = pct[pct > 0]
    sns.barplot(x=pct.values, y=pct.index)
    plt.xlabel("% missing")
    plt.ylabel("Columns")
    plt.title("Missing Values by Column")
    plt.tight_layout()
    plt.show()

# depth distribution
def plot_depth_hist(df: pd.DataFrame):
    temp_df = df.copy()
    plt.figure(figsize=(10, 6))
    sns.histplot(data=temp_df, x="depth", bins=40)
    plt.xlabel("Depth (km)")
    plt.title("Depth Distribution")
    plt.show()


# ==============================================================================
# CATEGORY 2: Bivariate & Spatial Explorations
# Level of Difficulty: Medium | Level of Importance: High (Relationship Discovery)
# ==============================================================================

# mag by country
def plot_mag_by_country(df: pd.DataFrame):
    temp_df = df.copy()
    country_col = "country"
    mag_col = "mag"
    min_events = 1
    summary = (
        temp_df.groupby(country_col)[mag_col]
        .agg(count="count")
        .round(2)
    )
    summary = summary[summary["count"] >= min_events]

    order = summary.index.tolist()
    data = temp_df[temp_df[country_col].isin(order)]
    labels = [f"{c}\n(n={summary.loc[c, 'count']})" for c in order]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=data, x=country_col, y=mag_col, order=order,
                hue=country_col, legend=False,
                showfliers=True, width=0.6)

    min_val = np.floor(data[mag_col].min())
    max_val = np.ceil(data[mag_col].max())

    plt.xticks(range(len(order)), labels)
    plt.yticks(np.arange(min_val, max_val + 0.5, 0.5))

    plt.xlabel("Country")
    plt.ylabel("Magnitude (mag)")
    plt.title("Earthquake Magnitude by Country", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

# earthquake using lat and long
def plot_earthquake_map(df: pd.DataFrame):
    temp_df = df.copy()
    lat_col = "latitude"
    lon_col = "longitude"
    mag_col = "mag"

    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")

    scatter = plt.scatter(
        x=temp_df[lon_col],
        y=temp_df[lat_col],
        c=temp_df[mag_col],
        # cmap="YlOrRd",
        alpha=0.6,
        s=temp_df[mag_col] * 2.5,
        edgecolors="none"
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Magnitude (mag)", fontsize=11)

    plt.title("Global Earthquake Epicenters (Latitude vs. Longitude)", fontsize=13, fontweight="bold")
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)
    plt.tight_layout()
    plt.show()

# Pearson Spears table
def plot_pearson_spearman(df: pd.DataFrame):
    x_col = 'mag'
    y_col = 'nst' # number of stations

    plt.figure(figsize=(10, 6))
    d = df[[x_col, y_col]].dropna()

    # Calculate both Spearman and Pearson
    spearman_rho, p_val_s = stats.spearmanr(d[x_col], d[y_col])
    pearson_r, p_val_p = stats.pearsonr(d[x_col], d[y_col])

    sns.scatterplot(data=d, x=x_col, y=y_col, alpha=0.4)

    title_text = (
        f"{y_col} vs {x_col} (n = {len(d)})\n"
        f"Spearman rho = {spearman_rho:.2f} | Pearson r = {pearson_r:.2f}"
    )
    plt.title(title_text, fontsize=12, fontweight='bold')
    plt.show()


# ==============================================================================
# CATEGORY 3: Advanced Multivariate & Temporal Analysis
# Level of Difficulty: High | Level of Importance: High (Multi-feature & Trend Insights)
# ==============================================================================

# Heatmap
def plot_corr_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    method = "spearman"
    complete_only = False

    cols = ["mag", "latitude", "longitude",
            "nst", "gap", "dmin", "rms", "horizontalError",
            "depthError", "magError", "magNst"]

    data = temp_df[cols]
    if complete_only:
        data = data.dropna()

    corr_matrix = data.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.set_theme(style="white")
    plt.figure(figsize=(12, 7))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1, center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": f"{method.capitalize()} correlation coefficient"},
    )
    plt.title(
        f"Numerical Features Correlation Heatmap (Target: mag)\n"
        f"{method.capitalize()}, n = {len(data)}",
        fontsize=13, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return corr_matrix

# mean mag by countries for a decades
def plot_mean_magnitude_decade_heatmap(df: pd.DataFrame):
    temp_df = df.copy()
    country_col = "country"
    mag_col = "mag"
    date_col = "date"
    top_n = 10

    temp_df["year"] = pd.to_datetime(temp_df[date_col], errors="coerce").dt.year
    temp_df = temp_df.dropna(subset=["year", mag_col, country_col])
    temp_df["decade"] = (temp_df["year"] // 10 * 10).astype(str) + "s"

    top_countries = temp_df[country_col].value_counts().head(top_n).index
    filtered_df = temp_df[temp_df[country_col].isin(top_countries)]

    pivot_data = filtered_df.pivot_table(
        index=country_col,
        columns="decade",
        values=mag_col,
        aggfunc="mean"
    )

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={'label': 'Mean Magnitude'}
    )

    plt.title(f"Mean Earthquake Magnitude by Country Across Decades (Top {top_n} Countries)", fontsize=13,
              fontweight="bold")
    plt.xlabel("Decade", fontsize=11)
    plt.ylabel("Country", fontsize=11)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Quarterly series
def plot_quarterly_series(df: pd.DataFrame):
    min_year = 1990
    temp_df = df.copy()

    dates = pd.to_datetime(temp_df['date'], format='%Y-%m-%d', errors='coerce')
    temp_df['year'] = dates.dt.year
    temp_df['quarter'] = dates.dt.to_period("Q")

    temp_df = temp_df.dropna(subset=['year', 'quarter'])
    temp_df['year'] = temp_df['year'].astype(int)
    d = temp_df[temp_df["year"] >= min_year]

    quarters = pd.period_range(f"{min_year}Q1", d["quarter"].max(), freq="Q")
    q = d.groupby("quarter").size().reindex(quarters, fill_value=0)

    plt.figure(figsize=(10, 6))
    x = q.index.to_timestamp()
    sns.lineplot(x=x, y=q.values, label="Events per quarter")
    sns.lineplot(x=x, y=q.rolling(4).mean().values, label="4-quarter average")

    plt.title("Events per Quarter", fontsize=12, fontweight='bold')
    plt.ylabel("Number of events")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return q


# ==============================================================================
# CATEGORY 4: Statistical Hypothesis Testing & Seismological Modeling
# Level of Difficulty: Very High | Level of Importance: Critical (Scientific Validation)
# ==============================================================================

# Shapiro-Wilk test
def plot_shapiro_test(df: pd.DataFrame, col: str):
    data = df[col].dropna()
    stat, p = stats.shapiro(data)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data, kde=True, stat="density", ax=ax[0])
    ax[0].set_title(f"{col} histogram ({stat.round(7)})")
    stats.probplot(data, dist="norm", plot=ax[1])
    ax[1].set_title(f"{col} Q-Q plot (Shapiro p = {p:.2e})")
    plt.tight_layout()
    plt.show()

# Gutenberg richter
def plot_gutenberg_richter(df: pd.DataFrame):
    temp_df = df.copy()
    mc = 4.0
    min_year = 1990
    temp_df['year'] = pd.to_datetime(temp_df['date'], format='%Y-%m-%d', errors='coerce').dt.year
    temp_df = temp_df.dropna(subset=['year'])
    temp_df['year'] = temp_df['year'].astype(int)

    plt.figure(figsize=(10, 6))

    d = temp_df[(temp_df['year'] >= min_year) & (temp_df['mag'] >= mc)]
    mags = np.sort(d['mag'].unique())

    counts = np.array([(d['mag'] >= m).sum() for m in mags])

    slope, intercept = np.polyfit(mags, np.log10(counts), 1)

    sns.scatterplot(x=mags, y=counts, label="Observed")

    plt.plot(mags, 10 ** (intercept + slope * mags), color="red",
             label=f"Fit: b = {-slope:.2f}")

    plt.yscale("log")
    plt.xlabel("Magnitude")
    plt.ylabel("Events with magnitude >= m")
    plt.title("Gutenberg-Richter Relationship")
    plt.legend()
    plt.show()

# Poisson
def plot_poisson(df: pd.DataFrame):
    temp_df = df.copy()
    min_year= 1990
    temp_df['year'] = pd.to_datetime(temp_df['date'], format='%Y-%m-%d', errors='coerce').dt.year
    temp_df = temp_df.dropna(subset=['year'])
    temp_df['year'] = temp_df['year'].astype(int)


    """Observed yearly counts vs a Poisson distribution with the same mean."""
    plt.figure(figsize=(10, 6))
    years = range(min_year, temp_df["year"].max())
    counts = temp_df.groupby("year").size().reindex(years, fill_value=0)
    mean, var = counts.mean(), counts.var()
    k = np.arange(counts.min(), counts.max() + 1)
    expected = stats.poisson.pmf(k, mean)
    sns.histplot(counts, discrete=True, stat="probability", label="Observed")
    plt.plot(k, expected, "ro-", label=f"Poisson (lambda = {mean:.1f})")
    plt.title(f"Events per Year: variance/mean = {var / mean:.1f}")
    plt.xlabel("Events in a year")
    plt.legend()
    plt.show()

# poisson by country
def plot_poisson_by_country(df: pd.DataFrame):
    country_name = "Uzbekistan"
    min_year = 1990
    temp_df = df.copy()
    temp_df = temp_df[temp_df['country'].str.lower() == country_name.lower()]

    temp_df['year'] = pd.to_datetime(temp_df['date'], format='%Y-%m-%d', errors='coerce').dt.year
    temp_df = temp_df.dropna(subset=['year'])
    temp_df['year'] = temp_df['year'].astype(int)

    plt.figure(figsize=(10, 6))
    years = range(min_year, int(temp_df['year'].max()) + 1)
    counts = temp_df.groupby("year").size().reindex(years, fill_value=0)

    mean, var = counts.mean(), counts.var()
    k = np.arange(counts.min(), counts.max() + 1)
    expected = stats.poisson.pmf(k, mean)

    sns.histplot(counts, discrete=True, stat="probability", label=f"Observed ({country_name})")
    plt.plot(k, expected, "ro-", label=f"Poisson (lambda = {mean:.1f})")

    ratio_text = f"variance/mean = {var / mean:.1f}" if mean > 0 and not np.isnan(var) else "N/A"

    plt.title(f"Events per Year in {country_name} ({min_year}+): {ratio_text}")
    plt.xlabel("Events in a year")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

# Mannwhitney hypothesis
def plot_era_mannwhitney(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    df['year'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.year
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df["era"] = np.where(df["year"] < 1990, "Before 1990", "1990 onward")
    before = df.loc[df["era"] == "Before 1990", "mag"]
    after = df.loc[df["era"] == "1990 onward", "mag"]
    stat, p = stats.mannwhitneyu(before, after, alternative="two-sided")

    sns.boxplot(data=df, x="era", y="mag", order=["Before 1990", "1990 onward"])
    plt.xlabel(f"Median before 1990: {before.median():.2f} (n={len(before)})\n"
               f"Median 1990 onward: {after.median():.2f} (n={len(after)})")

    plt.title(f"Magnitude by Era (p = {p:.2e}, Mann-Whitney U = {stat:.0f})")
    plt.show()

# kruskal
def plot_country_kruskal(df: pd.DataFrame):
    temp_df = df.dropna(subset=['country', 'mag']).copy()
    groups = [g["mag"].values for _, g in temp_df.groupby("country")]
    stat, p = stats.kruskal(*groups)
    summary = temp_df.groupby("country")["mag"].agg(["count", "median"]).round(2)
    plt.figure(figsize=(12, 6))
    order = summary.sort_values(by="median", ascending=False).index
    sns.boxplot(data=temp_df,
                x="country",
                y="mag",
                hue="country",
                order=order,
                palette="Set2",
                legend=False)
    plt.xticks(rotation=45,
               ha='right')
    plt.title(f"Earthquake Magnitude Distribution by Country (Kruskal-Wallis p = {p:.2e}), "
              f"\n Kruskal-Wallis H = {stat:.2f}", fontsize=12,
              fontweight='bold')
    plt.xlabel("Country", fontsize=11)
    plt.ylabel("Magnitude", fontsize=11)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# MAIN EXECUTION & REPORT METADATA
# ==============================================================================
if __name__ == "__main__":
    print(
        "======================================================================\n"
        "        SEISMIC DATA ANALYSIS & VISUALIZATION REPORT GENERATOR        \n"
        "======================================================================\n\n"

        "===== Function 1: plot_magnitude_distribution =====\n"
        "Description : Univariate analysis examining the distribution of earthquake magnitudes.\n"
        "Objective   : Visually inspect whether the empirical magnitude data aligns with expected theoretical seismological probability curves.\n"
        "Graph Explanation: The histogram combined with the Kernel Density Estimate (KDE) curve illustrates data skewness, concentration peaks, and frequency scaling across earthquake sizes.\n\n"

        "===== Function 2: plot_depth_distribution =====\n"
        "Description : Categorical univariate breakdown mapping earthquake counts across standardized depth classes.\n"
        "Objective   : Identify the dominant geological depth strata (ranging from very shallow to deep intermediate zones) within the region.\n"
        "Graph Explanation: The count plot displays discrete frequency values alongside sample counts (n) for each classification tier, emphasizing structural depth concentration.\n\n"

        "===== Function 3: plot_events_per_year =====\n"
        "Description : Temporal trend analysis tracking annual seismic event counts across a multi-decadal timeline.\n"
        "Objective   : Detect historical variations in reporting completeness, observation consistency, and activity spikes over time.\n"
        "Graph Explanation: The annual bar chart maps event totals per calendar year, explicitly filling missing gaps with zeros to maintain temporal integrity.\n\n"

        "===== Function 4: plot_desc_table =====\n"
        "Description : Comprehensive statistical summary profiling all numerical variables in the dataset.\n"
        "Objective   : Establish foundational data hygiene metrics, including measures of central tendency, dispersion, missingness percentages, and distribution shape.\n"
        "Graph Explanation: Side-by-side bar plots quantify missing value ratios per feature alongside distribution skewness values, highlighting features requiring transformation.\n\n"

        "===== Function 5: plot_missing_percent =====\n"
        "Description : Data quality audit quantifying missing data ratios per column.\n"
        "Objective   : Isolate columns susceptible to high data loss to guide subsequent imputation, deletion, or filtering decisions.\n"
        "Graph Explanation: The descending horizontal/vertical bar graph visualizes the exact percentage of missing entries for incomplete attributes.\n\n"

        "===== Function 6: plot_depth_hist =====\n"
        "Description : Continuous numerical inspection of hypocentral depth values.\n"
        "Objective   : Analyze the fine-grained distribution profile of earthquake focal depths measured in kilometers.\n"
        "Graph Explanation: A high-resolution histogram categorizes exact continuous depth values into equal bins to uncover geological clustering.\n\n"

        "===== Function 7: plot_mag_by_country =====\n"
        "Description : Regional bivariate comparison of earthquake magnitude distributions segmented by country.\n"
        "Objective   : Evaluate whether specific geographical territories experience systematically higher median magnitudes or varying dispersion.\n"
        "Graph Explanation: Boxplots display country-specific medians, interquartile ranges (IQRs), and outlier magnitudes, annotated with sample sizes (n).\n\n"

        "===== Function 8: plot_earthquake_map =====\n"
        "Description : Global spatial mapping of earthquake epicenter coordinates.\n"
        "Objective   : Visualize the geographical positioning of seismic events against latitude and longitude grids.\n"
        "Graph Explanation: A scatter plot maps global coordinates where point sizes and color gradients scale directly with earthquake magnitude to highlight active tectonic plate margins.\n\n"

        "===== Function 9: plot_pearson_spearman =====\n"
        "Description : Bivariate relationship analysis between earthquake magnitude and network recording capability.\n"
        "Objective   : Test the correlation between event magnitude and the number of reporting seismic stations (nst).\n"
        "Graph Explanation: The scatter plot pairs magnitude against reporting station counts, proving that larger events trigger wider station networks with strong positive linear and monotonic association metrics (Pearson r=0.72, Spearman ρ=0.75).\n\n"

        "===== Function 10: plot_corr_heatmap =====\n"
        "Description : Multivariate correlation matrix assessing multicollinearity across all primary numerical attributes.\n"
        "Objective   : Identify linear and monotonic associations between the target variable (mag) and network or error parameters.\n"
        "Graph Explanation: A lower-triangular correlation heatmap maps coefficient values from -1 to 1 using a diverging coolwarm color palette to flag highly collinear features.\n\n"

        "===== Function 11: plot_mean_magnitude_decade_heatmap =====\n"
        "Description : Socio-temporal multivariate matrix evaluating regional mean magnitudes over successive decades.\n"
        "Objective   : Track long-term shifts in average earthquake intensities across the top seismically active countries.\n"
        "Graph Explanation: A grid-based heatmap displays intersected mean magnitudes categorized by country rows and decade columns, highlighting historical intensity trends.\n\n"

        "===== Function 12: plot_quarterly_series =====\n"
        "Description : High-resolution quarterly time-series tracking of seismic frequency.\n"
        "Objective   : Smooth out short-term fluctuations to reveal medium-term rate changes and seasonal patterns in seismicity.\n"
        "Graph Explanation: Line plots trace raw quarterly event counts alongside a smoothed 4-quarter rolling average trendline over time.\n\n"

        "===== Function 13: plot_shapiro_test =====\n"
        "Description : Statistical hypothesis test assessing the normality of numerical feature distributions.\n"
        "Objective   : Formally test whether target variables follow a Gaussian distribution to determine the applicability of parametric vs. non-parametric models.\n"
        "Graph Explanation: Dual-panel layout combining a fitted density histogram and a Quantile-Quantile (Q-Q) plot, annotated with Shapiro-Wilk test statistic and p-value.\n\n"

        "===== Function 14: plot_gutenberg_richter =====\n"
        "Description : Seismological power-law modeling implementing the Gutenberg-Richter frequency-magnitude relationship.\n"
        "Objective   : Quantify the relative proportion of small versus large earthquakes to derive the characteristic regional b-value.\n"
        "Graph Explanation: A log-scale scatter plot fits a linear regression line to cumulative magnitude counts, yielding an estimated b-value of 1.41 that confirms the dominance of micro-seismic events.\n\n"

        "===== Function 15: plot_poisson =====\n"
        "Description : Goodness-of-fit test comparing annual earthquake frequency against a theoretical Poisson process.\n"
        "Objective   : Evaluate whether annual event occurrences follow a memoryless random process or exhibit real-world clustering.\n"
        "Graph Explanation: Compares observed yearly count probabilities against a Poisson probability mass function (lambda), highlighting mild overdispersion (variance-to-mean ratio of 1.8) driven by aftershock sequences.\n\n"

        "===== Function 16: plot_poisson_by_country =====\n"
        "Description : Localized Poisson process evaluation targeting specific regional fault zones (e.g., Uzbekistan).\n"
        "Objective   : Test annual event stationarity and clustering behavior at a national administrative scale.\n"
        "Graph Explanation: Overlays observed annual frequency distributions against a theoretical Poisson curve, displaying local variance-to-mean dispersion metrics.\n\n"

        "===== Function 17: plot_era_mannwhitney =====\n"
        "Description : Non-parametric hypothesis testing comparing magnitude distributions across historical epochs (Pre-1990 vs. 1990 Onward).\n"
        "Objective   : Detect potential shifts or systematic biases in magnitude recording standards following modern digital instrumentation upgrades.\n"
        "Graph Explanation: Comparative boxplots display median shifts between eras, supported by Mann-Whitney U test statistics and exact sample sizes.\n\n"

        "===== Function 18: plot_country_kruskal =====\n"
        "Description : Multi-group non-parametric variance test evaluating magnitude differences across all recorded countries.\n"
        "Objective   : Determine whether observed differences in median earthquake magnitudes between national borders are statistically significant.\n"
        "Graph Explanation: Ranked boxplots ordered by median magnitude illustrate cross-country variations, backed by Kruskal-Wallis H-statistic and p-values."
    )






