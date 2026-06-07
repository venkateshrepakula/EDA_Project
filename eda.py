import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/titanic.csv")

# -----------------------------
# Basic Information
# -----------------------------
print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# -----------------------------
# Gender Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.savefig("images/gender_distribution.png")
plt.close()

# -----------------------------
# Survival Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Distribution")
plt.savefig("images/survival_distribution.png")
plt.close()

# -----------------------------
# Age Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=10)
plt.title("Age Distribution")
plt.savefig("images/age_distribution.png")
plt.close()

# -----------------------------
# Passenger Class Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.savefig("images/passenger_class_distribution.png")
plt.close()

# -----------------------------
# Survival by Gender
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", hue="Sex", data=df)
plt.title("Survival by Gender")
plt.savefig("images/survival_by_gender.png")
plt.close()

# -----------------------------
# Correlation Heatmap
# -----------------------------
df2 = df.copy()

df2["Sex"] = df2["Sex"].map({
    "male": 0,
    "female": 1
})

corr = df2.corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.close()

print("\n==============================")
print("EDA COMPLETED SUCCESSFULLY")
print("All graphs saved in images folder")
print("==============================")
