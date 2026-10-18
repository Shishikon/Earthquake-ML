import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_earthquake_magnitudes(df: pd.DataFrame):
    if "mag" not in df.columns:
        print("Column 'mag' not found in DataFrame.")
        return

    plt.figure(figsize=(10, 5))
    sns.histplot(df["mag"].dropna(), kde=True, color="purple", bins=30)
    plt.title("Earthquake Magnitude Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Magnitude", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.tight_layout()
    plt.show()

def mag_by_country(df: pd.DataFrame):
    df = "working"
    return df