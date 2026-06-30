"""
=========================================================
Customer Churn Prediction System
Generate Project Charts

Part 1
=========================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# Create Results Folder
# -------------------------------------------------------

os.makedirs("results", exist_ok=True)

# -------------------------------------------------------
# Plot Style
# -------------------------------------------------------

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)

print("Dataset Loaded Successfully")

# =====================================================
# Figure 6.1
# Customer Churn Distribution
# =====================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Churn",
    palette="Set2"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "results/Figure_6_1_Customer_Churn_Distribution.png",
    dpi=300
)

plt.close()

# =====================================================
# Figure 6.2
# Gender Distribution
# =====================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="gender",
    palette="pastel"
)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "results/Figure_6_2_Gender_Distribution.png",
    dpi=300
)

plt.close()

# =====================================================
# Figure 6.3
# Contract Type Distribution
# =====================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Contract",
    palette="Set3"
)

plt.title("Contract Type Distribution")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    "results/Figure_6_3_Contract_Distribution.png",
    dpi=300
)

plt.close()

# =====================================================
# Figure 6.4
# Internet Service Distribution
# =====================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="InternetService",
    palette="viridis"
)

plt.title("Internet Service Distribution")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "results/Figure_6_4_Internet_Service.png",
    dpi=300
)

plt.close()

print("Figures 6.1 to 6.4 Generated Successfully")
# =====================================================
# Figure 6.5
# Monthly Charges Distribution
# =====================================================

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="MonthlyCharges",
    bins=30,
    kde=True,
    color="steelblue"
)

plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "results/Figure_6_5_Monthly_Charges_Distribution.png",
    dpi=300
)

plt.close()

# =====================================================
# Figure 6.6
# Tenure Distribution
# =====================================================

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="tenure",
    bins=30,
    kde=True,
    color="orange"
)

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "results/Figure_6_6_Tenure_Distribution.png",
    dpi=300
)

plt.close()

# =====================================================
# Figure 6.7
# Churn by Contract Type
# =====================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn",
    palette="Set2"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    "results/Figure_6_7_Churn_by_Contract.png",
    dpi=300
)

plt.close()

# =====================================================
# Figure 6.8
# Churn by Internet Service
# =====================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn",
    palette="coolwarm"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "results/Figure_6_8_Churn_by_Internet_Service.png",
    dpi=300
)

plt.close()

# =====================================================
# Figure 6.9
# Correlation Heatmap
# =====================================================

numeric_df = df.copy()

numeric_df["Churn"] = numeric_df["Churn"].map({
    "No": 0,
    "Yes": 1
})

numeric_df["SeniorCitizen"] = numeric_df["SeniorCitizen"].astype(int)

numeric_df["TotalCharges"] = pd.to_numeric(
    numeric_df["TotalCharges"],
    errors="coerce"
)

corr = numeric_df[[
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="RdYlBu",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "results/Figure_6_9_Correlation_Heatmap.png",
    dpi=300
)

plt.close()

print("Figures 6.5 to 6.9 Generated Successfully")
# =====================================================
# Figure 6.10
# Model Accuracy Comparison
# =====================================================

import shutil

print("\nPreparing Final Report Figures...")

if os.path.exists("results/accuracy_comparison.png"):

    shutil.copy(

        "results/accuracy_comparison.png",

        "results/Figure_6_10_Model_Accuracy_Comparison.png"

    )

    print("Figure 6.10 Generated")

# =====================================================
# Figure 6.11
# Confusion Matrix
# =====================================================

if os.path.exists("results/confusion_matrix.png"):

    shutil.copy(

        "results/confusion_matrix.png",

        "results/Figure_6_11_Confusion_Matrix.png"

    )

    print("Figure 6.11 Generated")

# =====================================================
# Figure 6.12
# ROC Curve
# =====================================================

if os.path.exists("results/roc_curve.png"):

    shutil.copy(

        "results/roc_curve.png",

        "results/Figure_6_12_ROC_Curve.png"

    )

    print("Figure 6.12 Generated")

# =====================================================
# Summary
# =====================================================

print("\n")
print("=" * 60)
print("ALL CHARTS GENERATED SUCCESSFULLY")
print("=" * 60)

print("""
Generated Figures

Figure 6.1  Customer Churn Distribution
Figure 6.2  Gender Distribution
Figure 6.3  Contract Type Distribution
Figure 6.4  Internet Service Distribution
Figure 6.5  Monthly Charges Distribution
Figure 6.6  Customer Tenure Distribution
Figure 6.7  Customer Churn by Contract Type
Figure 6.8  Customer Churn by Internet Service
Figure 6.9  Correlation Heatmap
Figure 6.10 Model Accuracy Comparison
Figure 6.11 Confusion Matrix
Figure 6.12 ROC Curve

All figures are available inside the results folder.
""")

print("=" * 60)