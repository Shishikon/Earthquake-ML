import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import stats
from scipy import stats


# 🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽 Univariate START 🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽

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
def plot_desc_table(df) -> pd.DataFrame:
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
def plot_depth_hist(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="depth", bins=40)
    plt.xlabel("Depth (km)")
    plt.title("Depth Distribution")
    plt.show()


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
def plot_poisson(df):
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



# 🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼 Univariate END 🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼





# 🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽 Bivariate START 🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽

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


# poisson by country
def plot_poisson_by_country(df):
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
def plot_era_mannwhitney(df):
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
def plot_country_kruskal(df):
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
# 🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼 Bivariate END 🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼



# 🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽 Multivariate START 🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽
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
def plot_quarterly_series(df):
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
# 🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼 Multivariate END 🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼







# 🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽 TESTING START 🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽

# 🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼 TESTING END 🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼🔼














if __name__ == "__main__":
    print(f"=====Function 1=====\n"
          f"plot_magnitude_distribution: "
          f"This function will show the distribution of variable mag, which \n"
          f"helps visually inspect whether it matches expected seismological curves.\n"
          
          f"\n=====Function 2=====\n"
          f"plot_corr_heatmap: "
          f"This function will show the correlation heatmap of variable mag, which \n"
          f"will calculate the correlation coefficients between mag and all other \n"
          f"numerical columns (like depth, network parameters, and error metrics) \n"
          f"to see what linear relationships exist.\n"
    
          f"\n=====Function 3=====\n"
          f"plot_mag_by_country: "
          f"The full statistical distribution for each country, including the median \n"
          f"(middle line), the interquartile range (the box), and any extreme outliers \n"
          f"(dots outside the whiskers).\n"

          f"\n=====Function 4=====\n"
          f"plot_desc_table: "
          f"Computes a comprehensive descriptive statistics table for numeric columns, \n"
          f"displays a straightforward visualization of missing % and skewness, \n"
          f"and returns the DataFrame.\n"


          f"\n=====Function 5=====\n"
          f"plot_pearson_spearman: "
          f"A strong positive correlation exists between earthquake magnitude and the, \n"
          f"number of reporting stations (Pearson r=0.72, Spearman ρ=0.75, n=409), \n"
          f"demonstrating that larger seismic events are successfully detected by \n"
          f"broader monitoring networks.\n"


          f"\n=====Function 6=====\n"
          f"plot_gutenberg_richter: "
          f"The Gutenberg-Richter frequency-magnitude distribution demonstrates a strong \n "
          f"inverse power-law relationship, yielding ab-value of 1.41. This indicates a rapid \n"
          f"decrease in event frequency as magnitude increases, confirming that smaller magnitude \n"
          f"events dominate the catalog.\n"


          f"\n=====Function 7=====\n"
          f"plot_poisson: "
          f"The yearly earthquake frequency deviates slightly from a pure Poisson model, \n"
          f"showing mild overdispersion with a variance-to-mean ratio of 1.8. This reflects \n "
          f"real-world seismic clustering, where major events trigger aftershock sequences \n"
          f"within the same annual timeframe.\n")












