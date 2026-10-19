import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# mag distribution histogram
def plot_magnitude_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")

    mag_data = df["mag"].dropna()

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


# Heatmap
def plot_corr_heatmap(df: pd.DataFrame, method: str = "spearman", complete_only: bool = False) -> pd.DataFrame:

    cols = ["mag", "latitude", "longitude",
            "nst", "gap", "dmin", "rms", "horizontalError",
            "depthError", "magError", "magNst"]

    data = df[cols]
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

# mag by country
def plot_mag_by_country(df: pd.DataFrame, country_col: str = "country",
        mag_col: str = "mag",
        min_events: int = 1) -> pd.DataFrame:

    summary = (
        df.groupby(country_col)[mag_col]
        .agg(count="count")
        .round(2)
    )
    summary = summary[summary["count"] >= min_events]

    order = summary.index.tolist()
    data = df[df[country_col].isin(order)]
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

    date_col = "date"
    start_year = 1928
    end_year = 2026

    temp_df = df.copy()
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

    # Thin out X-axis ticks to show every 5 years so it isn't crowded
    tick_positions = np.arange(0, len(full_range), 5)
    tick_labels = list(full_range)[::5]
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)

    plt.title(f"Earthquake Frequency Per Year ({start_year}–{end_year})", fontsize=13, fontweight="bold")
    plt.xlabel("Year", fontsize=11)
    plt.ylabel("Number of Events", fontsize=11)
    plt.tight_layout()
    plt.show()


# mean mag by countries for a decades
def plot_mean_magnitude_decade_heatmap(df: pd.DataFrame):
    country_col = "country"
    mag_col = "mag"
    date_col = "date"
    top_n = 10


    temp_df = df.copy()

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
          f"(dots outside the whiskers).\n")












