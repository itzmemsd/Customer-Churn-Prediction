"""
=========================================================
Customer Churn Prediction System
MBA AI & DS Mini Project
Author: Dr. M. Sasidharan

PART 1
=========================================================
"""

# =====================================================
# Import Libraries
# =====================================================

import os
import warnings
import joblib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# =====================================================
# Create folders
# =====================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

print("=" * 70)
print("CUSTOMER CHURN PREDICTION SYSTEM")
print("=" * 70)

df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

print("\nDataset Loaded Successfully")
print("Original Shape :", df.shape)

# =====================================================
# Data Cleaning
# =====================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)

df.drop(columns=["customerID"], inplace=True)

print("Shape After Cleaning :", df.shape)

# =====================================================
# Features and Target
# =====================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]

# =====================================================
# Split Dataset
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================================
# Numerical and Categorical Columns
# =====================================================

numerical_columns = [

    "SeniorCitizen",

    "tenure",

    "MonthlyCharges",

    "TotalCharges"

]

categorical_columns = [

    column

    for column in X.columns

    if column not in numerical_columns

]

# =====================================================
# Preprocessing Pipeline
# =====================================================

preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            StandardScaler(),

            numerical_columns

        ),

        (

            "cat",

            OneHotEncoder(

                handle_unknown="ignore"

            ),

            categorical_columns

        )

    ]

)

# =====================================================
# Models
# =====================================================

models = {

    "Logistic Regression":

        LogisticRegression(

            max_iter=1000,

            random_state=42

        ),

    "Decision Tree":

        DecisionTreeClassifier(

            random_state=42

        ),

    "Random Forest":

        RandomForestClassifier(

            n_estimators=200,

            random_state=42

        ),

    "Support Vector Machine":

        SVC(

            probability=True,

            random_state=42

        )

}

# =====================================================
# Variables
# =====================================================

results = []

best_model = None

best_model_name = ""

best_accuracy = 0

best_prediction = None

best_probability = None

print("\nPreprocessing Completed Successfully.")
print("=" * 70)
print("MODEL TRAINING")
print("=" * 70)
# =====================================================
# Train All Models Using Pipeline
# =====================================================

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(

        steps=[

            ("preprocessor", preprocessor),

            ("classifier", model)

        ]

    )

    # Train Model

    pipeline.fit(

        X_train,

        y_train

    )

    # Prediction

    prediction = pipeline.predict(

        X_test

    )

    probability = pipeline.predict_proba(

        X_test

    )[:, 1]

    # Evaluation

    accuracy = accuracy_score(

        y_test,

        prediction

    )

    precision = precision_score(

        y_test,

        prediction,

        pos_label="Yes"

    )

    recall = recall_score(

        y_test,

        prediction,

        pos_label="Yes"

    )

    f1 = f1_score(

        y_test,

        prediction,

        pos_label="Yes"

    )

    auc = roc_auc_score(

        (y_test == "Yes").astype(int),

        probability

    )

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": auc

    })

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")

    # Save Best Model

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = pipeline

        best_model_name = model_name

        best_prediction = prediction

        best_probability = probability

# =====================================================
# Model Comparison
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(

    by="Accuracy",

    ascending=False

).reset_index(drop=True)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df)

# Save Comparison

results_df.to_csv(

    "results/model_comparison.csv",

    index=False

)

# =====================================================
# Save Best Model
# =====================================================

joblib.dump(

    best_model,

    "models/best_model.pkl"

)

print("\nBest Model Saved Successfully")

print("Model Name :", best_model_name)

print(f"Accuracy   : {best_accuracy:.4f}")
# =====================================================
# Classification Report
# =====================================================

print("\nGenerating Classification Report...")

report = classification_report(
    y_test,
    best_prediction
)

with open(
    "results/classification_report.txt",
    "w"
) as file:
    file.write(report)

print("Classification Report Saved")

# =====================================================
# Confusion Matrix
# =====================================================

print("\nGenerating Confusion Matrix...")

cm = confusion_matrix(
    y_test,
    best_prediction,
    labels=["No", "Yes"]
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)

plt.title(f"Confusion Matrix ({best_model_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion Matrix Saved")

# =====================================================
# ROC Curve
# =====================================================

print("\nGenerating ROC Curve...")

y_test_binary = (y_test == "Yes").astype(int)

fpr, tpr, threshold = roc_curve(
    y_test_binary,
    best_probability
)

roc_score = roc_auc_score(
    y_test_binary,
    best_probability
)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"{best_model_name} (AUC={roc_score:.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/roc_curve.png",
    dpi=300
)

plt.close()

print("ROC Curve Saved")

# =====================================================
# Accuracy Comparison Chart
# =====================================================

print("\nGenerating Accuracy Comparison Chart...")

plt.figure(figsize=(9,5))

bars = plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.002,
        f"{height:.3f}",
        ha="center"
    )

plt.ylabel("Accuracy")
plt.title("Machine Learning Model Comparison")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    "results/accuracy_comparison.png",
    dpi=300
)

plt.close()

print("Accuracy Comparison Chart Saved")

# =====================================================
# Feature Importance (Random Forest only)
# =====================================================

if best_model_name == "Random Forest":

    print("\nGenerating Feature Importance...")

    classifier = best_model.named_steps["classifier"]

    encoder = best_model.named_steps["preprocessor"]

    feature_names = encoder.get_feature_names_out()

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": classifier.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    importance.to_csv(

        "results/feature_importance.csv",

        index=False

    )

    plt.figure(figsize=(10,6))

    plt.barh(

        importance["Feature"][:10],

        importance["Importance"][:10]

    )

    plt.gca().invert_yaxis()

    plt.title("Top 10 Important Features")

    plt.tight_layout()

    plt.savefig(

        "results/feature_importance.png",

        dpi=300

    )

    plt.close()

    print("Feature Importance Saved")

# =====================================================
# Final Summary
# =====================================================

print("\n")
print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Best Model : {best_model_name}")
print(f"Accuracy   : {best_accuracy:.4f}")

print("\nGenerated Files")

files = [

    "models/best_model.pkl",

    "results/model_comparison.csv",

    "results/classification_report.txt",

    "results/confusion_matrix.png",

    "results/roc_curve.png",

    "results/accuracy_comparison.png"

]

if best_model_name == "Random Forest":

    files.append("results/feature_importance.csv")
    files.append("results/feature_importance.png")

for file in files:

    print(f"✓ {file}")

print("\nYou can now build the Streamlit application.")
print("=" * 70)