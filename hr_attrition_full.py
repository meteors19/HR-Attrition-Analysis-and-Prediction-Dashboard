
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

# UPDATE THIS PATH to where your CSV is saved
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(f"✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
print(f"\nAttrition Distribution:")
print(df['Attrition'].value_counts())
print(f"\nAttrition Rate: {round(df['Attrition'].value_counts(normalize=True)['Yes']*100, 2)}%")

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# Basic info
print("\nDataset Info:")
print(f"  - Total Employees: {len(df)}")
print(f"  - Employees who Left: {df['Attrition'].value_counts()['Yes']}")
print(f"  - Missing Values: {df.isnull().sum().sum()}")
print(f"  - Duplicate Rows: {df.duplicated().sum()}")

# Drop useless columns
cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df.drop(columns=cols_to_drop, inplace=True)
print(f"\n✅ Dropped constant columns: {cols_to_drop}")

# ---- PLOT 1: Attrition by Department ----
plt.figure(figsize=(10, 5))
dept_attrition = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack()
dept_attrition['Yes'].sort_values(ascending=False).plot(kind='bar', color='#2E75B6', edgecolor='black')
plt.title('Attrition Rate by Department', fontsize=14, fontweight='bold')
plt.xlabel('Department')
plt.ylabel('Attrition Rate')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('notebooks/plot1_department_attrition.png', dpi=150)
plt.close()
print("\n✅ Plot 1 saved: Attrition by Department")

# Find top attrition department
top_dept = df.groupby('Department')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).sort_values(ascending=False)
print(f"\n📊 Department Attrition Rates:")
for dept, rate in top_dept.items():
    print(f"   {dept}: {round(rate, 1)}%")

# ---- PLOT 2: Overtime vs Attrition ----
plt.figure(figsize=(8, 5))
overtime_attrition = df.groupby('OverTime')['Attrition'].value_counts(normalize=True).unstack()
overtime_attrition['Yes'].plot(kind='bar', color=['#1F4E79', '#2E75B6'], edgecolor='black')
plt.title('Attrition Rate: Overtime vs No Overtime', fontsize=14, fontweight='bold')
plt.xlabel('Overtime')
plt.ylabel('Attrition Rate')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('notebooks/plot2_overtime_attrition.png', dpi=150)
plt.close()
print("✅ Plot 2 saved: Overtime vs Attrition")

ot_yes = round(df[df['OverTime'] == 'Yes']['Attrition'].value_counts(normalize=True)['Yes'] * 100, 1)
ot_no = round(df[df['OverTime'] == 'No']['Attrition'].value_counts(normalize=True)['Yes'] * 100, 1)
print(f"\n📊 Overtime Insight:")
print(f"   Employees WITH overtime: {ot_yes}% attrition")
print(f"   Employees WITHOUT overtime: {ot_no}% attrition")
print(f"   → Overtime employees are {round(ot_yes/ot_no, 1)}x more likely to leave")

# ---- PLOT 3: Job Satisfaction vs Attrition ----
plt.figure(figsize=(8, 5))
sat_attrition = df.groupby('JobSatisfaction')['Attrition'].value_counts(normalize=True).unstack()
sat_attrition['Yes'].plot(kind='bar', color='#C00000', edgecolor='black')
plt.title('Attrition Rate by Job Satisfaction Level', fontsize=14, fontweight='bold')
plt.xlabel('Job Satisfaction (1=Low, 4=High)')
plt.ylabel('Attrition Rate')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('notebooks/plot3_satisfaction_attrition.png', dpi=150)
plt.close()
print("✅ Plot 3 saved: Job Satisfaction vs Attrition")

# ---- PLOT 4: Age Distribution ----
plt.figure(figsize=(10, 5))
df[df['Attrition'] == 'Yes']['Age'].hist(alpha=0.7, label='Left', color='#C00000', bins=20)
df[df['Attrition'] == 'No']['Age'].hist(alpha=0.7, label='Stayed', color='#2E75B6', bins=20)
plt.title('Age Distribution: Attrition vs Retained', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/plot4_age_attrition.png', dpi=150)
plt.close()
print("✅ Plot 4 saved: Age Distribution")

# ---- PLOT 5: Correlation Heatmap ----
plt.figure(figsize=(14, 10))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/plot5_correlation_heatmap.png', dpi=150)
plt.close()
print("✅ Plot 5 saved: Correlation Heatmap")

# ============================================================
# STEP 3: PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Data Preprocessing")
print("=" * 60)

df_model = df.copy()

# Encode target
df_model['Attrition'] = df_model['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categorical columns
le = LabelEncoder()
cat_cols = df_model.select_dtypes(include='object').columns.tolist()
print(f"Encoding {len(cat_cols)} categorical columns: {cat_cols}")

for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

print("✅ Label encoding complete")

# Features and target
X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train/Test split: {X_train.shape[0]} train | {X_test.shape[0]} test")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Feature scaling complete")

# ============================================================
# STEP 4: MODEL TRAINING & COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Model Training & Comparison")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Use scaled data for LR, unscaled for tree models
    X_tr = X_train_scaled if name == "Logistic Regression" else X_train
    X_te = X_test_scaled if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')

    results[name] = {
        'model': model,
        'accuracy': acc,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    print(f"\n📊 {name}:")
    print(f"   Accuracy:     {round(acc * 100, 2)}%")
    print(f"   ROC-AUC:      {round(auc, 4)}")
    print(f"   CV Accuracy:  {round(cv_scores.mean() * 100, 2)}% ± {round(cv_scores.std() * 100, 2)}%")

# Best model
best_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_name]['model']
print(f"\n🏆 Best Model: {best_name} with {round(results[best_name]['accuracy']*100, 2)}% accuracy")

# ---- PLOT 6: Model Comparison ----
plt.figure(figsize=(10, 5))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] * 100 for m in model_names]
aucs = [results[m]['auc'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35
plt.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#2E75B6')
plt.bar(x + width/2, [a * 100 for a in aucs], width, label='AUC × 100', color='#1F4E79')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, model_names)
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/plot6_model_comparison.png', dpi=150)
plt.close()
print("\n✅ Plot 6 saved: Model Comparison")

# ---- PLOT 7: Confusion Matrix (Best Model) ----
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Left'],
            yticklabels=['Stayed', 'Left'])
plt.title(f'Confusion Matrix — {best_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('notebooks/plot7_confusion_matrix.png', dpi=150)
plt.close()
print("✅ Plot 7 saved: Confusion Matrix")

# ---- PLOT 8: ROC Curve ----
plt.figure(figsize=(8, 6))
for name in results:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={round(results[name]['auc'], 3)})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/plot8_roc_curve.png', dpi=150)
plt.close()
print("✅ Plot 8 saved: ROC Curve")

# ---- PLOT 9: Feature Importance ----
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind='barh', color='#2E75B6', edgecolor='black')
plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('notebooks/plot9_feature_importance.png', dpi=150)
plt.close()
print("✅ Plot 9 saved: Feature Importance")

print(f"\n📊 Top 5 Attrition Drivers:")
for feat, score in top_features.head(5).items():
    print(f"   {feat}: {round(score, 4)}")

# ---- Classification Report ----
print(f"\n📋 Classification Report — {best_name}:")
print(classification_report(y_test, results[best_name]['y_pred'],
                            target_names=['Stayed', 'Left']))

# ============================================================
# STEP 5: SAVE MODEL & SCALER
# ============================================================
print("=" * 60)
print("STEP 5: Saving Model & Scaler")
print("=" * 60)

os.makedirs('model', exist_ok=True)

# Save best model
with open('model/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save Random Forest specifically (for feature importance in app)
with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save scaler
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature columns
with open('model/feature_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("✅ model/best_model.pkl saved")
print("✅ model/rf_model.pkl saved")
print("✅ model/scaler.pkl saved")
print("✅ model/feature_columns.pkl saved")

print("\n" + "=" * 60)
print("✅ PHASE 1 & 2 COMPLETE — EDA + ML Pipeline Done!")
print("📁 Check notebooks/ folder for all plots")
print("📁 Check model/ folder for saved models")
print("=" * 60)
print(f"\n🎯 YOUR REAL RESUME NUMBERS:")
print(f"   Dataset size: {len(df):,} records")
print(f"   Best Model: {best_name}")
print(f"   Accuracy: {round(results[best_name]['accuracy']*100, 2)}%")
print(f"   ROC-AUC: {round(results[best_name]['auc'], 4)}")
print(f"   Top attrition driver: {top_features.index[0]}")
