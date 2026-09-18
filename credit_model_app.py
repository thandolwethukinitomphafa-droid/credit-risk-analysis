"""
DataQuest 2026: Building Interpretable Credit Models
Interactive Credit Risk Analysis & Decision Support Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit
st.set_page_config(page_title="Credit Risk Analysis", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# DATA LOADING & CACHING
# ============================================================================

@st.cache_data
def load_data():
    """Load and preprocess the loan book data"""
    df = pd.read_csv('loan_book.csv')
    return df

@st.cache_resource
def compute_woe_iv(df, feature, target):
    """
    Compute Weight of Evidence (WoE) and Information Value (IV) for a feature.
    
    WoE measures the strength of a feature's relationship to the target.
    IV measures the predictive power of a feature.
    """
    # Handle missing values
    df_clean = df[[feature, target]].dropna()
    
    # For continuous variables, use quantile binning
    if df_clean[feature].dtype in ['float64', 'int64'] and df_clean[feature].nunique() > 10:
        df_clean['bin'] = pd.qcut(df_clean[feature], q=5, duplicates='drop')
    else:
        df_clean['bin'] = df_clean[feature]
    
    # Calculate contingency table
    ct = pd.crosstab(df_clean['bin'], df_clean[target])
    
    # Calculate proportions
    ct['total_events'] = ct[1]
    ct['total_non_events'] = ct[0]
    
    total_events = df_clean[target].sum()
    total_non_events = (1 - df_clean[target]).sum()
    
    ct['pct_events'] = ct[1] / total_events
    ct['pct_non_events'] = ct[0] / total_non_events
    
    # Avoid log(0)
    ct['pct_events'] = ct['pct_events'].replace(0, 0.0001)
    ct['pct_non_events'] = ct['pct_non_events'].replace(0, 0.0001)
    
    # Calculate WoE and IV
    ct['woe'] = np.log(ct['pct_events'] / ct['pct_non_events'])
    ct['iv_component'] = (ct['pct_events'] - ct['pct_non_events']) * ct['woe']
    
    iv = ct['iv_component'].sum()
    
    return ct, iv

@st.cache_data
def prepare_model_data(df):
    """Prepare features for modeling"""
    df = df.copy()
    
    # Feature engineering - create meaningful transformations
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 75], 
                             labels=['21-30', '31-40', '41-50', '50+'])
    
    # Income groups
    df['income_group'] = pd.qcut(df['annual_income'].fillna(df['annual_income'].median()), 
                                 q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Debt-to-income ratio categories
    df['dti_category'] = pd.cut(df['dti_ratio'], bins=[0, 0.15, 0.30, 0.45, 1.0],
                               labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Credit utilization categories
    df['credit_util_cat'] = pd.cut(df['credit_utilisation_pct'], 
                                   bins=[0, 30, 60, 90, 100],
                                   labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Delinquency history
    df['has_delinquency'] = (df['num_delinquencies_2yr'] > 0).astype(int)
    df['delinquency_count_cat'] = pd.cut(df['num_delinquencies_2yr'], 
                                         bins=[-1, 0, 2, 5, 100],
                                         labels=['None', '1-2', '3-5', '6+'])
    
    # Recent credit inquiries
    df['has_recent_inquiry'] = (df['num_hard_inquiries_6mo'] > 0).astype(int)
    
    # Account age
    df['account_age_years'] = df['months_since_oldest_account'] / 12
    df['account_age_cat'] = pd.cut(df['account_age_years'], 
                                   bins=[0, 5, 10, 20, 400],
                                   labels=['<5yr', '5-10yr', '10-20yr', '20+yr'])
    
    # Credit history quality
    df['accounts_current_pct'] = df['pct_accounts_current']
    df['credit_quality'] = pd.cut(df['pct_accounts_current'], 
                                  bins=[0, 70, 85, 95, 101],
                                  labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    # Handle missing values
    df['employment_length_years'] = df['employment_length_years'].fillna(df['employment_length_years'].median())
    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())
    df['num_open_accounts'] = df['num_open_accounts'].fillna(df['num_open_accounts'].median())
    
    # Encode categorical variables
    df['home_ownership_encoded'] = pd.factorize(df['home_ownership'].str.upper())[0]
    df['is_phone_verified'] = df['phone_verified'].astype(int)
    df['email_is_corporate'] = (df['email_domain_type'] == 'corporate').astype(int)
    
    return df

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🏦 Credit Risk Analysis & Modeling Platform")
    st.markdown("**DataQuest 2026** | Building Interpretable Credit Models")
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(" loan_book.csv not found. Please ensure the file is in the same directory.")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🔍 EDA Explorer",
        "🤖 Model Development",
        "📈 Model Performance",
        "💼 Business Dashboard"
    ])
    
    # ========================================================================
    # TAB 1: DATA OVERVIEW
    # ========================================================================
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Default Rate", f"{df['default_flag'].mean()*100:.1f}%")
        with col3:
            st.metric("Features", df.shape[1])
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("📋 Column Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numeric Features:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                st.write(f"- {col}")
        
        with col2:
            st.write("**Categorical Features:**")
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in cat_cols:
                st.write(f"- {col}")
        
        st.subheader(" Data Quality Report")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Missing Values by Column:**")
            missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
            missing_pct = missing_pct[missing_pct > 0]
            if len(missing_pct) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                missing_pct.plot(kind='barh', ax=ax, color='coral')
                ax.set_xlabel('Missing %')
                st.pyplot(fig)
            else:
                st.info(" No missing values in target variable (default_flag)")
        
        with col2:
            st.write("**Train/Test Split:**")
            set_dist = df['set'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            set_dist.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
            ax.set_title('Data Distribution (Train/Test)')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        st.subheader(" Default Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            df['default_flag'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
            ax.set_xticklabels(['No Default', 'Default'], rotation=0)
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        with col2:
            st.write("**Default Statistics:**")
            st.write(f"- Non-Defaulters: {(df['default_flag']==0).sum():,} ({(df['default_flag']==0).mean()*100:.1f}%)")
            st.write(f"- Defaulters: {(df['default_flag']==1).sum():,} ({(df['default_flag']==1).mean()*100:.1f}%)")
            st.write(f"- Imbalance Ratio: {(df['default_flag']==0).sum() / (df['default_flag']==1).sum():.1f}:1")
    
    # ========================================================================
    # TAB 2: EDA EXPLORER
    # ========================================================================
    with tab2:
        st.header(" Exploratory Data Analysis")
        
        eda_mode = st.radio("Select Analysis Mode:", 
                           ["Univariate Analysis", "Bivariate Analysis", "Credit Metrics"])
        
        if eda_mode == "Univariate Analysis":
            st.subheader("Single Variable Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ['applicant_id_hash', 'default_flag', 'set']]
            
            selected_feature = st.selectbox("Select a feature:", numeric_cols)
            
            if selected_feature:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Distribution of {selected_feature}**")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(df[selected_feature].dropna(), bins=50, color='skyblue', edgecolor='black')
                    ax.set_xlabel(selected_feature)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
                
                with col2:
                    st.write(f"**{selected_feature} by Default Status**")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    df.boxplot(column=selected_feature, by='default_flag', ax=ax)
                    ax.set_xticklabels(['No Default', 'Default'])
                    ax.set_ylabel(selected_feature)
                    st.pyplot(fig)
                
                # Calculate WoE and IV
                st.write("**Weight of Evidence (WoE) & Information Value (IV)**")
                try:
                    ct, iv = compute_woe_iv(df, selected_feature, 'default_flag')
                    st.write(f"Information Value (IV): **{iv:.4f}**")
                    st.write("*IV Interpretation: <0.02=weak, 0.02-0.1=fair, 0.1-0.3=good, >0.3=strong*")
                    st.dataframe(ct[['woe', 'iv_component']])
                except Exception as e:
                    st.warning(f"Could not calculate WoE/IV: {e}")
        
        elif eda_mode == "Bivariate Analysis":
            st.subheader("Two-Variable Analysis")
            
            all_cols = [c for c in df.columns if c not in ['applicant_id_hash', 'default_flag', 'set', 'application_date']]
            
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("First feature:", all_cols, index=0)
            with col2:
                feature2 = st.selectbox("Second feature:", all_cols, index=1)
            
            if feature1 and feature2:
                # Check if both are numeric
                is_numeric1 = df[feature1].dtype in [np.float64, np.int64]
                is_numeric2 = df[feature2].dtype in [np.float64, np.int64]
                
                if is_numeric1 and is_numeric2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(df[feature1], df[feature2], 
                                       c=df['default_flag'], cmap='RdYlGn_r', alpha=0.5)
                    ax.set_xlabel(feature1)
                    ax.set_ylabel(feature2)
                    plt.colorbar(scatter, ax=ax, label='Default')
                    st.pyplot(fig)
                
                else:
                    # Create crosstab heatmap
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ct = pd.crosstab(
                        df[feature1].astype(str),
                        df[feature2].astype(str),
                        values=df['default_flag'],
                        aggfunc='mean'
                    )
                    sns.heatmap(ct, annot=True, fmt='.2%', cmap='RdYlGn_r', ax=ax)
                    ax.set_title(f'Default Rate: {feature1} vs {feature2}')
                    st.pyplot(fig)
        
        else:  # Credit Metrics
            st.subheader("Credit Risk Metrics")
            
            credit_features = ['age', 'annual_income', 'credit_utilisation_pct', 
                             'num_delinquencies_2yr', 'dti_ratio', 
                             'pct_accounts_current', 'employment_length_years']
            
            st.write("**Information Value for Key Features:**")
            iv_results = []
            for feat in credit_features:
                try:
                    _, iv = compute_woe_iv(df, feat, 'default_flag')
                    iv_results.append({'Feature': feat, 'IV': iv})
                except:
                    pass
            
            iv_df = pd.DataFrame(iv_results).sort_values('IV', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(iv_df['Feature'], iv_df['IV'], color='steelblue')
            ax.set_xlabel('Information Value')
            ax.set_title('Predictive Power of Credit Features')
            st.pyplot(fig)
            
            st.dataframe(iv_df, use_container_width=True)
    
    # ========================================================================
    # TAB 3: MODEL DEVELOPMENT
    # ========================================================================
    with tab3:
        st.header(" Model Development")
        
        # Prepare data
        df_model = prepare_model_data(df)
        
        st.subheader("Feature Engineering Process")
        st.info("""
        **Transformations Applied:**
        - Age, Income, DTI Ratio: Binned into interpretable categories
        - Credit Utilization: 4-point scale (Low/Medium/High/Very High)
        - Delinquency: Binary flag + count categories
        - Credit Quality: Based on % accounts current
        - Account Age: Converted to years + categories
        - Employment: Median imputation for missing values
        """)
        
        # Select features for modeling
        numeric_features = ['age', 'annual_income', 'employment_length_years', 
                          'num_open_accounts', 'total_revolving_balance',
                          'credit_utilisation_pct', 'months_since_oldest_account',
                          'num_hard_inquiries_6mo', 'loan_amount', 'interest_rate',
                          'dti_ratio', 'pct_accounts_current', 'months_at_current_address',
                          'is_phone_verified', 'email_is_corporate',
                          'num_delinquencies_2yr', 'has_delinquency', 'has_recent_inquiry']
        
        # Prepare X and y
        X = df_model[numeric_features].copy()
        y = df_model['default_flag'].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median(numeric_only=True))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        st.write(f"**Training Set Size:** {len(X_train):,}")
        st.write(f"**Test Set Size:** {len(X_test):,}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression
        log_reg = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
        log_reg.fit(X_train_scaled, y_train)
        
        st.subheader(" Model Equation (Linear Predictor η)")
        
        # Display coefficients
        coef_df = pd.DataFrame({
            'Feature': numeric_features,
            'Coefficient': log_reg.coef_[0],
            'Abs_Coefficient': np.abs(log_reg.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)
        
        st.write(f"**Intercept (β₀):** {log_reg.intercept_[0]:.6f}")
        st.dataframe(coef_df, use_container_width=True)
        
        st.write("**Logistic Regression Equation:**")
        st.latex(r"P(\text{Default}) = \frac{1}{1 + e^{-\eta}}")
        st.write("where η = " + " + ".join([f"{c:.4f}×{f}" if c > 0 else f"{c:.4f}×{f}" 
                                           for f, c in zip(numeric_features, log_reg.coef_[0])])[:200] + "...")
        
        # Store model for other tabs
        st.session_state.log_reg = log_reg
        st.session_state.scaler = scaler
        st.session_state.X_train_scaled = X_train_scaled
        st.session_state.X_test_scaled = X_test_scaled
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.X_test = X_test
        st.session_state.numeric_features = numeric_features
    
    # ========================================================================
    # TAB 4: MODEL PERFORMANCE
    # ========================================================================
    with tab4:
        st.header("Model Performance Evaluation")
        
        if 'log_reg' not in st.session_state:
            st.warning(" Please train the model in the 'Model Development' tab first.")
            return
        
        log_reg = st.session_state.log_reg
        X_test_scaled = st.session_state.X_test_scaled
        y_test = st.session_state.y_test
        
        # Get predictions
        y_pred = log_reg.predict(X_test_scaled)
        y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        st.subheader("Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AUC Score", f"{auc_score:.4f}", 
                     help="Baseline: 0.68, LightGBM: 0.82")
        with col2:
            st.metric("Precision", f"{precision:.4f}", 
                     help="Of predicted defaults, how many actually defaulted?")
        with col3:
            st.metric("Recall", f"{recall:.4f}",
                     help="Of actual defaults, how many did we catch?")
        with col4:
            st.metric("F1 Score", f"{f1:.4f}",
                     help="Harmonic mean of precision and recall")
        
        # ROC Curve
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_ylabel('True')
            ax.set_xlabel('Predicted')
            ax.set_xticklabels(['No Default', 'Default'])
            ax.set_yticklabels(['No Default', 'Default'])
            st.pyplot(fig)
        
        st.subheader("Prediction Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(y_pred_proba[y_test==0], bins=50, alpha=0.6, label='No Default', color='green')
        ax.hist(y_pred_proba[y_test==1], bins=50, alpha=0.6, label='Default', color='red')
        ax.set_xlabel('Predicted Probability of Default')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Performance vs Baseline")
        metrics_comparison = pd.DataFrame({
            'Model': ['Baseline Logistic Regression', 'Our Improved Model', 'LightGBM (ceiling)'],
            'AUC': [0.68, auc_score, 0.82]
        })
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(metrics_comparison['Model'], metrics_comparison['AUC'], 
               color=['lightcoral', 'lightgreen', 'skyblue'])
        ax.set_xlabel('AUC Score')
        ax.set_xlim([0.6, 0.85])
        for i, v in enumerate(metrics_comparison['AUC']):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center')
        st.pyplot(fig)
    
    # ========================================================================
    # TAB 5: BUSINESS DASHBOARD
    # ========================================================================
    with tab5:
        st.header(" Business Decision Dashboard")
        
        if 'log_reg' not in st.session_state:
            st.warning(" Please train the model in the 'Model Development' tab first.")
            return
        
        log_reg = st.session_state.log_reg
        X_test_scaled = st.session_state.X_test_scaled
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        # Get predictions
        y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
        
        st.subheader(" Approval Strategy Analysis")
        
        # Allow user to set different thresholds
        threshold = st.slider("Decision Threshold (Default Probability):", 
                             min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        # Calculate metrics at this threshold
        y_pred_custom = (y_pred_proba >= threshold).astype(int)
        approved = (y_pred_custom == 0).sum()
        rejected = (y_pred_custom == 1).sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Approved Applications", f"{approved:,}", 
                     help=f"{approved/len(y_test)*100:.1f}% of population")
        with col2:
            st.metric("Rejected Applications", f"{rejected:,}",
                     help=f"{rejected/len(y_test)*100:.1f}% of population")
        with col3:
            approval_rate = approved / len(y_test) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        # Risk metrics at selected threshold
        st.subheader("Risk Profile at Selected Threshold")
        
        # Among approved, how many actually default?
        approved_defaults = ((y_pred_custom == 0) & (y_test == 1)).sum()
        approved_non_defaults = ((y_pred_custom == 0) & (y_test == 0)).sum()
        
        actual_default_rate_in_approved = approved_defaults / approved if approved > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Default Rate (Approved)", 
                     f"{actual_default_rate_in_approved*100:.2f}%",
                     help="% of approved applicants expected to default")
        with col2:
            precision_at_threshold = approved_non_defaults / approved if approved > 0 else 0
            st.metric("Precision @ Threshold",
                     f"{precision_at_threshold*100:.2f}%",
                     help="% of approvals that don't default")
        with col3:
            recall_at_threshold = approved_non_defaults / (y_test == 0).sum()
            st.metric("Recall @ Threshold",
                     f"{recall_at_threshold*100:.2f}%",
                     help="% of non-defaulters we approved")
        
        # Volume vs Risk Tradeoff
        st.subheader(" Volume vs Risk Tradeoff")
        
        thresholds = np.arange(0, 1.01, 0.05)
        volume_data = []
        
        for thresh in thresholds:
            approved_at_thresh = (y_pred_proba >= thresh).sum()
            approval_pct = approved_at_thresh / len(y_test)
            
            if approved_at_thresh > 0:
                defaults_in_approved = ((y_pred_proba >= thresh) & (y_test == 1)).sum()
                default_rate = defaults_in_approved / approved_at_thresh
            else:
                default_rate = 0
            
            volume_data.append({
                'Threshold': thresh,
                'Approval Rate': approval_pct * 100,
                'Default Rate': default_rate * 100
            })
        
        volume_df = pd.DataFrame(volume_data)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Decision Threshold')
        ax1.set_ylabel('Approval Rate (%)', color=color)
        ax1.plot(volume_df['Threshold'], volume_df['Approval Rate'], 
                color=color, marker='o', linewidth=2, label='Approval Rate')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Default Rate Among Approved (%)', color=color)
        ax2.plot(volume_df['Threshold'], volume_df['Default Rate'], 
                color=color, marker='s', linewidth=2, label='Default Rate')
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax1.axvline(x=threshold, color='green', linestyle='--', alpha=0.5, 
                   label=f'Selected Threshold ({threshold:.2f})')
        
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        st.pyplot(fig)
        
        # Decision Matrix
        st.subheader("💡 Approval Recommendations")
        
        recommendations = f"""
        **At threshold {threshold:.2f}:**
        
        - **Approval Rate:** {approval_rate:.1f}% of applications
        - **Approved Volume:** {approved:,} customers
        - **Rejected Volume:** {rejected:,} customers
        - **Expected Defaults in Approved:** {approved_defaults} accounts
        - **Default Rate Among Approved:** {actual_default_rate_in_approved*100:.2f}%
        
        **Business Implications:**
        """
        
        if approval_rate > 80:
            recommendations += "\n-  HIGH VOLUME: Very permissive policy; high default risk"
        elif approval_rate > 60:
            recommendations += "\n-  BALANCED: Moderate approval rate with reasonable risk"
        else:
            recommendations += "\n-  CONSERVATIVE: Strict policy; low default risk but reduced volume"
        
        st.info(recommendations)
        
        # Customer segmentation by risk
        st.subheader(" Customer Segmentation by Risk")
        
        X_test['pred_prob'] = y_pred_proba
        X_test['actual_default'] = y_test.values
        
        X_test['risk_segment'] = pd.cut(X_test['pred_prob'], 
                                       bins=[0, 0.25, 0.5, 0.75, 1.0],
                                       labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
        
        segmentation = X_test.groupby('risk_segment').agg({
            'actual_default': ['count', 'sum', 'mean']
        }).round(3)
        
        segmentation.columns = ['Volume', 'Defaults', 'Default Rate']
        st.dataframe(segmentation, use_container_width=True)

if __name__ == "__main__":
    main()
