import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

df = sns.load_dataset("titanic")
df["age"] = df["age"].fillna(df["age"].median())


def save_and_show(filename):
    """Utility helper to layout, save, display, and close plots cleanly."""
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()


# 1. Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="sex", hue="survived", palette="Set2")
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
save_and_show("survival_by_gender.png")

# 2. Age Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["age"], kde=True, bins=20, color="skyblue")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
save_and_show("age_distribution.png")

# 3. Fare Distribution by Class
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="class", y="fare", palette="Set3")
plt.title("Ticket Fare Distribution Across Passenger Classes")
plt.xlabel("Class")
plt.ylabel("Fare ($)")
save_and_show("fare_by_class.png")

# 4. Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=["float64", "int64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
save_and_show("correlation_heatmap.png")