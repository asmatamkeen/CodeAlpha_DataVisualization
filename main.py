import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Load built-in dataset directly from Seaborn (no download needed!)
df = sns.load_dataset("titanic")

# Display first 5 rows
print("Dataset Sample:")
print(df.head())

# Clean missing values for age
df["age"] = df["age"].fillna(df["age"].median())

# ---------------------------------------------------------
# Visual 1: Categorical Bar Chart (Survival Count by Gender)
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="sex", hue="survived", palette="Set2")
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig("survival_by_gender.png")  # Saves image locally
plt.show()

# ---------------------------------------------------------
# Visual 2: Histogram / Distribution (Age Distribution)
# ---------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.histplot(df["age"], kde=True, bins=20, color="skyblue")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()

# ---------------------------------------------------------
# Visual 3: Box Plot (Ticket Fare by Class)
# ---------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="class", y="fare", palette="Set3")
plt.title("Ticket Fare Distribution Across Passenger Classes")
plt.xlabel("Class")
plt.ylabel("Fare ($)")
plt.tight_layout()
plt.savefig("fare_by_class.png")
plt.show()

# ---------------------------------------------------------
# Visual 4: Correlation Heatmap
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=["float64", "int64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()