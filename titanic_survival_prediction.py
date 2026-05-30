"""
╔══════════════════════════════════════════════════════════════╗
║         TITANIC SURVIVAL PREDICTION — ML PROJECT            ║
║──────────────────────────────────────────────────────────────║
║  Author     : RITUSMITA DUTTA                                 ║
║  Department : Electronics & Communication Engineering (ECE-B)║
║  Institute  : RCC Institute of Information Technology        ║
║  Tools      : Python · Pandas · NumPy · Seaborn · Sklearn   ║
║  Objective  : Predict passenger survival using ML            ║
╚══════════════════════════════════════════════════════════════╝

PROBLEM STATEMENT:
    The RMS Titanic sank on April 15, 1912 after colliding with an
    iceberg. Of 2,224 passengers, only 710 survived. This project
    builds a Machine Learning model to predict whether a passenger
    survived, based on features like age, gender, and ticket class.

DATASET:
    Source  : datasciencedojo/datasets (GitHub)
    Records : 891 passengers | 12 features
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

warnings.filterwarnings("ignore")          # suppress future-version warnings
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})

print("=" * 60)
print("  TITANIC SURVIVAL PREDICTION  |  ML PROJECT")
print("=" * 60)
print("✔  All libraries imported successfully\n")


# ──────────────────────────────────────────────
# 2. LOAD DATASET
# ──────────────────────────────────────────────
URL = ("https://raw.githubusercontent.com/"
       "datasciencedojo/datasets/master/titanic.csv")

df = pd.read_csv(URL)

print(f"✔  Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Survival rate   →  {df['Survived'].mean()*100:.1f}% survived\n")


# ──────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────
print("─" * 60)
print("  SECTION 1 — Exploratory Data Analysis")
print("─" * 60)

# Missing value summary
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_report = pd.DataFrame({"Missing Count": missing,
                                "Missing (%)": missing_pct})
missing_report = missing_report[missing_report["Missing Count"] > 0]
print("\n📋 Missing Values:")
print(missing_report.to_string())

# Key statistics
print(f"\n📊 Key Statistics:")
print(f"   Average Age      : {df['Age'].mean():.1f} years")
print(f"   Average Fare     : £{df['Fare'].mean():.2f}")
print(f"   Female Survival  : {df[df['Sex']=='female']['Survived'].mean()*100:.1f}%")
print(f"   Male Survival    : {df[df['Sex']=='male']['Survived'].mean()*100:.1f}%")
print(f"   1st Class Survival: {df[df['Pclass']==1]['Survived'].mean()*100:.1f}%")
print(f"   3rd Class Survival: {df[df['Pclass']==3]['Survived'].mean()*100:.1f}%")


# ──────────────────────────────────────────────
# 4. VISUALISATION  (all 5 plots in one figure)
# ──────────────────────────────────────────────
print("\n─" * 60)
print("  SECTION 2 — Data Visualisation")
print("─" * 60)

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Titanic Dataset — Exploratory Data Analysis",
             fontsize=16, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

SURVIVED_COLORS = ["#E74C3C", "#2ECC71"]   # red = no, green = yes

# ── Plot 1: Overall survival count ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
survival_counts = df['Survived'].value_counts()
bars = ax1.bar(["Did Not Survive", "Survived"],
               survival_counts.values,
               color=SURVIVED_COLORS, edgecolor="white", linewidth=1.2,
               width=0.55)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 6,
             str(int(bar.get_height())),
             ha="center", va="bottom", fontweight="bold", fontsize=11)
ax1.set_title("Overall Survival Count", fontweight="bold")
ax1.set_ylabel("Number of Passengers")
ax1.set_ylim(0, survival_counts.max() * 1.15)
ax1.grid(axis="y", alpha=0.4)

# ── Plot 2: Survival by Gender ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
gender_survival = df.groupby(['Sex', 'Survived']).size().unstack()
gender_survival.plot(kind='bar', ax=ax2, color=SURVIVED_COLORS,
                     edgecolor='white', linewidth=1.2, rot=0, width=0.6)
ax2.set_title("Survival by Gender", fontweight="bold")
ax2.set_xlabel("Gender")
ax2.set_ylabel("Count")
ax2.legend(["Did Not Survive", "Survived"], loc="upper right", fontsize=9)
ax2.grid(axis="y", alpha=0.4)

# ── Plot 3: Survival by Passenger Class ─────────────────────
ax3 = fig.add_subplot(gs[0, 2])
class_survival = df.groupby(['Pclass', 'Survived']).size().unstack()
class_survival.plot(kind='bar', ax=ax3, color=SURVIVED_COLORS,
                    edgecolor='white', linewidth=1.2, rot=0, width=0.6)
ax3.set_title("Survival by Passenger Class", fontweight="bold")
ax3.set_xlabel("Class (1=First  2=Second  3=Third)")
ax3.set_ylabel("Count")
ax3.legend(["Did Not Survive", "Survived"], loc="upper right", fontsize=9)
ax3.grid(axis="y", alpha=0.4)

# ── Plot 4: Age Distribution by Survival ────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
for survived, color, label in zip([0, 1],
                                   SURVIVED_COLORS,
                                   ["Did Not Survive", "Survived"]):
    subset = df[df["Survived"] == survived]["Age"].dropna()
    ax4.hist(subset, bins=28, alpha=0.65, color=color,
             edgecolor="white", label=label)
ax4.axvline(df["Age"].median(), color="navy", linestyle="--",
            linewidth=1.5, label=f"Median Age ({df['Age'].median():.0f})")
ax4.set_title("Age Distribution by Survival Outcome", fontweight="bold")
ax4.set_xlabel("Age (years)")
ax4.set_ylabel("Number of Passengers")
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# ── Plot 5: Survival Rate by Class & Gender (heatmap) ───────
ax5 = fig.add_subplot(gs[1, 2])
pivot = df.pivot_table(values="Survived",
                       index="Pclass", columns="Sex", aggfunc="mean")
pivot = pivot * 100   # convert to percentage
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
            linewidths=0.5, ax=ax5, cbar_kws={"label": "Survival %"},
            vmin=0, vmax=100)
ax5.set_title("Survival Rate % by Class & Gender", fontweight="bold")
ax5.set_xlabel("Gender")
ax5.set_ylabel("Passenger Class")
ax5.set_yticklabels(["1st", "2nd", "3rd"], rotation=0)

plt.savefig("titanic_eda_dashboard.png", bbox_inches="tight")
plt.show()
print("\n✔  EDA dashboard saved  →  titanic_eda_dashboard.png")


# ──────────────────────────────────────────────
# 5. DATA PREPROCESSING
# ──────────────────────────────────────────────
print("\n─" * 60)
print("  SECTION 3 — Data Preprocessing")
print("─" * 60)

df_model = df.copy()

# Fill missing values (no inplace to avoid FutureWarning)
df_model["Age"]      = df_model["Age"].fillna(df_model["Age"].median())
df_model["Embarked"] = df_model["Embarked"].fillna(df_model["Embarked"].mode()[0])

# Encode categorical features to numeric
df_model["Sex"]      = df_model["Sex"].map({"male": 0, "female": 1})
df_model["Embarked"] = df_model["Embarked"].map({"S": 0, "C": 1, "Q": 2})

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df_model[FEATURES]
y = df_model["Survived"]

print(f"✔  Missing values imputed (Age → median, Embarked → mode)")
print(f"✔  Categorical encoding complete (Sex, Embarked)")
print(f"✔  Feature matrix shape  :  {X.shape}")
print(f"✔  Target variable shape :  {y.shape}")
print(f"   Features used         :  {FEATURES}")


# ──────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✔  Train-Test Split (80/20 stratified)")
print(f"   Training samples  :  {len(X_train)}")
print(f"   Testing  samples  :  {len(X_test)}")


# ──────────────────────────────────────────────
# 7. MODEL TRAINING
# ──────────────────────────────────────────────
print("\n─" * 60)
print("  SECTION 4 — Model Training & Evaluation")
print("─" * 60)

model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
model.fit(X_train, y_train)

print("\n✔  Logistic Regression model trained")
print(f"   Solver     : lbfgs")
print(f"   Max Iter   : 500")


# ──────────────────────────────────────────────
# 8. MODEL EVALUATION
# ──────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Cross-validation for robustness
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print(f"\n📊 Model Performance:")
print(f"   Test Accuracy       : {accuracy * 100:.2f}%")
print(f"   Cross-Val Accuracy  : {cv_scores.mean()*100:.2f}% "
      f"(± {cv_scores.std()*100:.2f}%)")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["Did Not Survive", "Survived"]))

# ── Confusion Matrix + Feature Importance (side by side) ────
fig2, (ax_cm, ax_fi) = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Model Evaluation — Logistic Regression",
              fontsize=14, fontweight="bold")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"],
            ax=ax_cm, annot_kws={"size": 14})
ax_cm.set_title(f"Confusion Matrix  |  Accuracy: {accuracy*100:.2f}%",
                fontweight="bold")
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("Actual Label")

# Feature Importance (coefficients)
coefs = pd.Series(np.abs(model.coef_[0]), index=FEATURES).sort_values()
colors = ["#3498DB" if c < coefs.median() else "#2ECC71" for c in coefs]
coefs.plot(kind="barh", ax=ax_fi, color=colors, edgecolor="white")
ax_fi.set_title("Feature Importance (|Coefficient|)", fontweight="bold")
ax_fi.set_xlabel("Absolute Coefficient Value")
ax_fi.set_ylabel("Feature")
ax_fi.grid(axis="x", alpha=0.4)

plt.tight_layout()
plt.savefig("titanic_model_evaluation.png", bbox_inches="tight")
plt.show()
print("\n✔  Evaluation plots saved  →  titanic_model_evaluation.png")


# ──────────────────────────────────────────────
# 9. LIVE PREDICTION DEMO
# ──────────────────────────────────────────────
print("\n─" * 60)
print("  SECTION 5 — Live Prediction Demo")
print("─" * 60)

test_cases = [
    {"label": "3rd-class male, age 22, alone",
     "data": [3, 0, 22, 0, 0, 7.25, 0]},
    {"label": "1st-class female, age 35, with spouse",
     "data": [1, 1, 35, 1, 0, 83.47, 1]},
    {"label": "2nd-class male, age 14, with parents",
     "data": [2, 0, 14, 0, 2, 26.00, 0]},
]

print()
for case in test_cases:
    passenger = pd.DataFrame([case["data"]], columns=FEATURES)
    pred   = model.predict(passenger)[0]
    prob   = model.predict_proba(passenger)[0][1]
    result = "✅ SURVIVED" if pred == 1 else "❌ DID NOT SURVIVE"
    print(f"  [{case['label']}]")
    print(f"   Prediction : {result}  |  Survival Probability: {prob*100:.1f}%\n")


# ──────────────────────────────────────────────
# 10. PROJECT SUMMARY
# ──────────────────────────────────────────────
print("=" * 60)
print("  PROJECT SUMMARY")
print("=" * 60)
print(f"  Dataset          :  Titanic — 891 passengers")
print(f"  Algorithm        :  Logistic Regression")
print(f"  Features Used    :  {len(FEATURES)}  {FEATURES}")
print(f"  Test Accuracy    :  {accuracy * 100:.2f}%")
print(f"  Cross-Val Score  :  {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"  Plots Saved      :  titanic_eda_dashboard.png")
print(f"                      titanic_model_evaluation.png")
print("=" * 60)
print("  ✅  Project Complete!")
print("=" * 60)

