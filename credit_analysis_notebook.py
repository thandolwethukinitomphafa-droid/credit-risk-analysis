"""
DataQuest 2026: Credit Risk Modeling - Complete Analysis Notebook
Executable Python script for reproducible analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, precision_score,
    recall_score, f1_score, classification_report, auc
)
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("DATAQUEST 2026: BUILDING INTERPRETABLE CREDIT MODELS")
print("="*80)

# ============================================================================
# 1. DATA LOADING & EXPLORATION
# ============================================================================

print("\n" + "="*80)
print("1. LOADING AND EXPLORING DATA")
print("="*80)

df = pd.read_csv('loan_book.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"\nColumn Names and Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDefault Distribution:\n{df['default_flag'].value_counts()}")
print(f"Default Rate: {df['default_flag'].mean():.2%}")

# ============================================================================
# 2. DATA QUALITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("2. DATA QUALITY REPORT")
print("="*80)

missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_summary = missing_pct[missing_pct > 0]
print("\nMissing Value Summary:")
print(missing_summary)

print("\nDescriptive Statistics:")
print(df.describe())

# ============================================================================
# 3. UNIVARIATE ANALYSIS & WOE/IV CALCULATION
# ============================================================================

print("\n" + "="*80)
print("3. UNIVARIATE ANALYSIS - WEIGHT OF EVIDENCE (WoE) & INFORMATION VALUE (IV)")
print("="*80)

def calculate_woe_iv(df, feature, target, bins=5):
    """
    Calculate WoE and IV for a feature
    
    WoE = ln(% of Events / % of Non-Events)
    IV = Σ[(% of Events - % of Non-Events) × WoE]
    """
    # Clean data
    df_clean = df[[feature, target]].dropna()
    
    # Bin continuous variables
    if df_clean[feature].dtype in ['float64', 'int64'] and df_clean[feature].nunique() > bins:
        df_clean['bin'] = pd.qcut(df_clean[feature], q=bins, duplicates='drop')
    else:
        df_clean['bin'] = df_clean[feature]
    
    # Create contingency table
    ct = pd.crosstab(df_clean['bin'], df_clean[target], margins=True)
    
    # Calculate proportions
    total_events = df_clean[target].sum()
    total_non_events = (df_clean[target] == 0).sum()
    
    woe_iv = []
    
    for idx in ct.index[:-1]:  # Exclude margins
        events = ct.loc[idx, 1]
        non_events = ct.loc[idx, 0]
        
        pct_events = events / total_events if total_events > 0 else 0
        pct_non_events = non_events / total_non_events if total_non_events > 0 else 0
        
        # Avoid log(0)
        pct_events = max(pct_events, 0.0001)
        pct_non_events = max(pct_non_events, 0.0001)
        
        woe = np.log(pct_events / pct_non_events)
        iv_component = (pct_events - pct_non_events) * woe
        
        woe_iv.append({
            'Bin': idx,
            'Count': events + non_events,
            'Events': events,
            'Non-Events': non_events,
            'WoE': woe,
            'IV_Component': iv_component
        })
    
    woe_iv_df = pd.DataFrame(woe_iv)
    total_iv = woe_iv_df['IV_Component'].sum()
    
    return woe_iv_df, total_iv

# Calculate IV for key features
key_features = [
    'age', 'annual_income', 'employment_length_years',
    'credit_utilisation_pct', 'num_delinquencies_2yr',
    'dti_ratio', 'pct_accounts_current', 'num_open_accounts',
    'months_since_oldest_account', 'interest_rate'
]

print("\nInformation Value (IV) for Key Features:")
print("IV Interpretation: <0.02=weak, 0.02-0.1=fair, 0.1-0.3=good, >0.3=strong\n")

iv_results = []
for feature in key_features:
    try:
        woe_df, iv = calculate_woe_iv(df, feature, 'default_flag')
        iv_results.append({'Feature': feature, 'IV': iv})
        print(f"{feature:30s}: IV = {iv:.4f}")
    except Exception as e:
        print(f"{feature:30s}: Error - {e}")

iv_summary = pd.DataFrame(iv_results).sort_values('IV', ascending=False)
print("\n" + iv_summary.to_string(index=False))

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("4. FEATURE ENGINEERING & TRANSFORMATION")
print("="*80)

df_engineered = df.copy()

print("\nApplying transformations...")

# 1. Age groups
df_engineered['age_group'] = pd.cut(df_engineered['age'], 
                                    bins=[20, 30, 40, 50, 75],
                                    labels=['21-30', '31-40', '41-50', '50+'],
                                    include_lowest=True)

# 2. Income groups
df_engineered['income_group'] = pd.qcut(
    df_engineered['annual_income'].fillna(df_engineered['annual_income'].median()),
    q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop'
)

# 3. DTI categories
df_engineered['dti_category'] = pd.cut(
    df_engineered['dti_ratio'],
    bins=[0, 0.15, 0.30, 0.45, 1.0],
    labels=['Low', 'Medium', 'High', 'Very High'],
    include_lowest=True
)

# 4. Credit utilization
df_engineered['credit_util_cat'] = pd.cut(
    df_engineered['credit_utilisation_pct'],
    bins=[0, 30, 60, 90, 100],
    labels=['Low', 'Medium', 'High', 'Very High'],
    include_lowest=True
)

# 5. Delinquency features
df_engineered['has_delinquency'] = (df_engineered['num_delinquencies_2yr'] > 0).astype(int)
df_engineered['delinquency_count'] = pd.cut(
    df_engineered['num_delinquencies_2yr'],
    bins=[-1, 0, 2, 5, 100],
    labels=['None', '1-2', '3-5', '6+'],
    include_lowest=True
)

# 6. Recent inquiries
df_engineered['has_recent_inquiry'] = (df_engineered['num_hard_inquiries_6mo'] > 0).astype(int)

# 7. Account age in years
df_engineered['account_age_years'] = df_engineered['months_since_oldest_account'] / 12
df_engineered['account_age_cat'] = pd.cut(
    df_engineered['account_age_years'],
    bins=[0, 5, 10, 20, 500],
    labels=['<5yr', '5-10yr', '10-20yr', '20+yr'],
    include_lowest=True
)

# 8. Credit quality
df_engineered['credit_quality'] = pd.cut(
    df_engineered['pct_accounts_current'],
    bins=[0, 70, 85, 95, 101],
    labels=['Poor', 'Fair', 'Good', 'Excellent'],
    include_lowest=True
)

# 9. Categorical encodings
df_engineered['home_ownership_encoded'] = pd.factorize(
    df_engineered['home_ownership'].str.upper()
)[0]

# 10. Behavioral flags
df_engineered['is_phone_verified'] = df_engineered['phone_verified'].astype(int)
df_engineered['email_is_corporate'] = (df_engineered['email_domain_type'] == 'corporate').astype(int)

# Handle missing values
df_engineered['employment_length_years'] = df_engineered['employment_length_years'].fillna(
    df_engineered['employment_length_years'].median()
)
df_engineered['annual_income'] = df_engineered['annual_income'].fillna(
    df_engineered['annual_income'].median()
)
df_engineered['num_open_accounts'] = df_engineered['num_open_accounts'].fillna(
    df_engineered['num_open_accounts'].median()
)

print("✓ Feature engineering complete")
print(f"  - Original features: {len(df.columns)}")
print(f"  - Engineered features: {len(df_engineered.columns)}")

# ============================================================================
# 5. MODEL PREPARATION & TRAINING
# ============================================================================

print("\n" + "="*80)
print("5. MODEL DEVELOPMENT - LOGISTIC REGRESSION")
print("="*80)

# Select features for modeling
numeric_features = [
    'age', 'annual_income', 'employment_length_years',
    'num_open_accounts', 'total_revolving_balance',
    'credit_utilisation_pct', 'months_since_oldest_account',
    'num_hard_inquiries_6mo', 'loan_amount', 'interest_rate',
    'dti_ratio', 'pct_accounts_current', 'months_at_current_address',
    'is_phone_verified', 'email_is_corporate',
    'num_delinquencies_2yr', 'has_delinquency', 'has_recent_inquiry'
]

X = df_engineered[numeric_features].copy()
y = df_engineered['default_flag'].copy()

# Ensure no NaN values
X = X.fillna(X.median())

print(f"\nFeatures selected: {len(numeric_features)}")
print(f"Training set shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}\n")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train default rate: {y_train.mean():.2%}")
print(f"Test default rate: {y_test.mean():.2%}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
print("\nTraining logistic regression...")
log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    penalty='l2'
)
log_reg.fit(X_train_scaled, y_train)
print("✓ Training complete")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("6. MODEL EVALUATION & PERFORMANCE")
print("="*80)

# Generate predictions
y_pred_train = log_reg.predict(X_train_scaled)
y_pred_proba_train = log_reg.predict_proba(X_train_scaled)[:, 1]

y_pred_test = log_reg.predict(X_test_scaled)
y_pred_proba_test = log_reg.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
print("\n--- TRAINING SET METRICS ---")
train_auc = roc_auc_score(y_train, y_pred_proba_train)
train_precision = precision_score(y_train, y_pred_train)
train_recall = recall_score(y_train, y_pred_train)
train_f1 = f1_score(y_train, y_pred_train)

print(f"AUC:       {train_auc:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall:    {train_recall:.4f}")
print(f"F1 Score:  {train_f1:.4f}")

print("\n--- TEST SET METRICS ---")
test_auc = roc_auc_score(y_test, y_pred_proba_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

print(f"AUC:       {test_auc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1 Score:  {test_f1:.4f}")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred_test, 
                          target_names=['No Default', 'Default']))

print("\n--- BASELINE COMPARISON ---")
print(f"Baseline AUC (old model):     0.68")
print(f"Our Model AUC:                {test_auc:.4f} ({'+' if test_auc > 0.68 else ''}{(test_auc-0.68)*100:.2f}%)")
print(f"LightGBM Ceiling (reference): 0.82")

# ============================================================================
# 7. MODEL COEFFICIENTS & INTERPRETATION
# ============================================================================

print("\n" + "="*80)
print("7. MODEL COEFFICIENTS & INTERPRETATION")
print("="*80)

print(f"\nIntercept (β₀): {log_reg.intercept_[0]:.6f}\n")

coef_df = pd.DataFrame({
    'Feature': numeric_features,
    'Coefficient': log_reg.coef_[0],
    'Abs_Coefficient': np.abs(log_reg.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("Top Risk Factors (by absolute coefficient):")
print(coef_df.head(10).to_string(index=False))

print("\n\nInterpretation Guide:")
print("-" * 80)
print("Positive coefficient (β > 0)  : INCREASES default risk")
print("Negative coefficient (β < 0)  : DECREASES default risk (protective)")
print("Coefficient of 0.10           : 1-unit increase ↑ log-odds by 0.10")
print("                              : ↑ relative odds by e^0.10 ≈ 10.5%")
print("-" * 80)

# ============================================================================
# 8. BUSINESS DECISION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("8. BUSINESS DECISION FRAMEWORK")
print("="*80)

print("\nApproval Rate vs Risk Tradeoff:")
print("-" * 80)

thresholds = np.arange(0.3, 1.01, 0.1)
decisions = []

for thresh in thresholds:
    approved = (y_pred_proba_test >= thresh).sum()
    rejected = (y_pred_proba_test < thresh).sum()
    approval_rate = approved / len(y_test)
    
    if approved > 0:
        defaults_in_approved = ((y_pred_proba_test >= thresh) & (y_test == 1)).sum()
        default_rate_approved = defaults_in_approved / approved
    else:
        default_rate_approved = 0
    
    decisions.append({
        'Threshold': f"{thresh:.2f}",
        'Approved': approved,
        'Rejected': rejected,
        'Approval %': f"{approval_rate*100:.1f}%",
        'Default % (Approved)': f"{default_rate_approved*100:.2f}%"
    })

decisions_df = pd.DataFrame(decisions)
print(decisions_df.to_string(index=False))

print("\n\nRECOMMENDED APPROVAL STRATEGY (Threshold = 0.50):")
print("-" * 80)
thresh_50 = 0.50
approved_50 = (y_pred_proba_test >= thresh_50).sum()
defaults_50 = ((y_pred_proba_test >= thresh_50) & (y_test == 1)).sum()
default_rate_50 = defaults_50 / approved_50 if approved_50 > 0 else 0

print(f"Approval Rate:           {approved_50/len(y_test)*100:.1f}%")
print(f"Approved Applications:   {approved_50:,}")
print(f"Rejected Applications:   {len(y_test) - approved_50:,}")
print(f"Expected Defaults:       {defaults_50:,} accounts")
print(f"Default Rate (Approved): {default_rate_50*100:.2f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("9. GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ROC Curve
ax = axes[0, 0]
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)

ax.plot(fpr_train, tpr_train, label=f'Train AUC = {train_auc:.4f}', linewidth=2)
ax.plot(fpr_test, tpr_test, label=f'Test AUC = {test_auc:.4f}', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Confusion Matrix
ax = axes[0, 1]
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_ylabel('True')
ax.set_xlabel('Predicted')
ax.set_title('Confusion Matrix (Test Set)')
ax.set_xticklabels(['No Default', 'Default'])
ax.set_yticklabels(['No Default', 'Default'])

# 3. Prediction Distribution
ax = axes[1, 0]
ax.hist(y_pred_proba_test[y_test == 0], bins=50, alpha=0.6, 
        label='No Default', color='green', edgecolor='black')
ax.hist(y_pred_proba_test[y_test == 1], bins=50, alpha=0.6,
        label='Default', color='red', edgecolor='black')
ax.set_xlabel('Predicted Probability of Default')
ax.set_ylabel('Frequency')
ax.set_title('Prediction Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Top Coefficients
ax = axes[1, 1]
top_coef = coef_df.head(10)
colors = ['red' if x > 0 else 'green' for x in top_coef['Coefficient']]
ax.barh(top_coef['Feature'], top_coef['Coefficient'], color=colors)
ax.set_xlabel('Coefficient Value')
ax.set_title('Top 10 Risk Factors')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_performance.png")

# ============================================================================
# SUMMARY & CONCLUSIONS
# ============================================================================

print("\n" + "="*80)
print("10. SUMMARY & CONCLUSIONS")
print("="*80)

print(f"""
MODEL PERFORMANCE SUMMARY
═════════════════════════════════════════════════════════════════════════════

Test Set AUC:                {test_auc:.4f}
  └─ Baseline Improvement:   {(test_auc - 0.68)*100:+.2f}% vs 0.68
  └─ Distance from Ceiling:  {(0.82 - test_auc)*100:.2f}% below LightGBM (0.82)

Precision (Specificity):     {test_precision:.4f}
  └─ Of rejected applicants, {test_precision*100:.1f}% would have defaulted

Recall (Sensitivity):        {test_recall:.4f}
  └─ Of defaulters, we identify {test_recall*100:.1f}%

INTERPRETATION
═════════════════════════════════════════════════════════════════════════════

Key Protective Factors (Negative Coefficients):
  1. Higher income
  2. Longer employment history
  3. Good account history (% current)
  4. More accounts open
  5. Phone verification

Key Risk Factors (Positive Coefficients):
  1. Recent delinquencies
  2. High debt-to-income ratio
  3. High credit utilization
  4. Recent hard inquiries
  5. Lower credit quality

BUSINESS RECOMMENDATIONS
═════════════════════════════════════════════════════════════════════════════

Recommended Policy:  50th percentile threshold (predict_proba ≥ 0.50)
  └─ Approve:        {approved_50:,} applications ({approved_50/len(y_test)*100:.1f}%)
  └─ Reject:         {len(y_test)-approved_50:,} applications
  └─ Expected Default Rate: {default_rate_50*100:.2f}%

Next Steps:
  1. Validate on out-of-sample data
  2. Implement scoring system based on model
  3. Monitor actual vs predicted defaults monthly
  4. Audit for disparate impact by protected characteristics
  5. Refit model quarterly with new data

═════════════════════════════════════════════════════════════════════════════
""")

print("\n✓ Analysis complete!")
print("Files generated:")
print("  - model_performance.png")
print("  - This notebook output")
