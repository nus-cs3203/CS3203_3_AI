import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder for saving plots
os.makedirs("plots", exist_ok=True)

# Load datasets
datasets = {
    "Normal": pd.read_csv("files/sentiment_results_normal.csv"),
    "Singlish": pd.read_csv("files/sentiment_results_singlish_robust.csv"),
    "Custom": pd.read_csv("files/sentiment_results_custom.csv"),
    "Advanced": pd.read_csv("files/sentiment_results_advanced.csv"),
}

# Ensure 'title_with_desc_score' is numeric
for name, df in datasets.items():
    df["title_with_desc_score"] = pd.to_numeric(df["title_with_desc_score"], errors="coerce")
    df.dropna(subset=["title_with_desc_score"], inplace=True)

# Save summary statistics
summary_df = pd.DataFrame({name: df["title_with_desc_score"].describe() for name, df in datasets.items()})
summary_df.to_csv("plots/sentiment_summary.csv")
print("Summary statistics saved to plots/sentiment_summary.csv")

# Pie charts for sentiment label distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, df) in zip(axes, datasets.items()):
    if "title_with_desc_label" in df.columns:
        label_counts = df["title_with_desc_label"].value_counts(normalize=True).reindex(["positive", "neutral", "negative"], fill_value=0)
        ax.pie(label_counts, labels=label_counts.index, autopct="%.1f%%", colors=["blue", "gray", "red"])
        ax.set_title(f"Sentiment Distribution - {name}")

plt.tight_layout()
plt.savefig("plots/sentiment_piecharts.png")
plt.close()

# Generate histograms for each dataset
for name, df in datasets.items():
    plt.figure(figsize=(8, 5))
    sns.histplot(df["title_with_desc_score"], bins=20, kde=True, color="blue")
    plt.title(f"Histogram of Sentiment Scores - {name}")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.savefig(f"plots/histogram_{name.lower()}.png")
    plt.close()

print("All plots and results are saved in the 'plots/' folder.")
