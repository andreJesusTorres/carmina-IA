# =============================================================================
# EXPLORATORY DATA ANALYSIS AND DATA CLEANING
# Essential approach based on EPFL course standards
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# 1. LOAD AND EXPLORE DATA STRUCTURE
# =============================================================================

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS AND DATA CLEANING")
print("=" * 60)

# Load the dataset
data_df = pd.read_csv("house-prices.csv")
print(f"Dataset shape: {data_df.shape}")
print(f"Memory usage: {data_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Extract target variable
y = data_df['SalePrice'].values
X_df = data_df.drop('SalePrice', axis=1)

print(f"\nTarget variable (SalePrice) range: ${y.min():,.0f} - ${y.max():,.0f}")

# =============================================================================
# 2. COMPREHENSIVE FEATURE EXPLORATION
# =============================================================================

print("\n" + "=" * 60)
print("02. Feature exploration")
print("=" * 60)

# Identify data types
dtype_info = X_df.dtypes
numerical_features = dtype_info[dtype_info.isin(['int64', 'float64'])].index.tolist()
categorical_features = dtype_info[dtype_info == 'object'].index.tolist()

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Check for missing values
missing_values = X_df.isnull().sum()
features_with_missing = missing_values[missing_values > 0]

if len(features_with_missing) > 0:
    print(f"\nFeatures with missing values: {len(features_with_missing)}")
    print("Top 10 features with most missing values:")
    print(features_with_missing.head(10).to_string())
else:
    print("\nNo missing values found")

# =============================================================================
# 3. TARGET VARIABLE ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("03. Target variable analysis")
print("=" * 60)

print(f"Target variable (SalePrice) analysis:")
print(f"Mean: ${y.mean():,.0f}")
print(f"Median: ${np.median(y):,.0f}")
print(f"Std: ${y.std():,.0f}")
print(f"Skewness: {stats.skew(y):.3f}")
print(f"Kurtosis: {stats.kurtosis(y):.3f}")

# Visualize target distribution
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.hist(y, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('SalePrice ($)')
plt.ylabel('Frequency')
plt.title('Distribution of SalePrice')

plt.subplot(1, 4, 2)
y_log = np.log(y)
plt.hist(y_log, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('log(SalePrice)')
plt.ylabel('Frequency')
plt.title('Distribution of log(SalePrice)')

plt.subplot(1, 4, 3)
plt.boxplot(y)
plt.ylabel('SalePrice ($)')
plt.title('Boxplot of SalePrice')

plt.subplot(1, 4, 4)
stats.probplot(y_log, dist="norm", plot=plt)
plt.title('Q-Q Plot (log-transformed)')

plt.tight_layout()
plt.show()

print("\nOBSERVATIONS:")
print("- SalePrice distribution is right-skewed (skewness > 0)")
print("- Log transformation makes the distribution more normal")
print("- Q-Q plot shows log-transformed data follows normal distribution better")
print("DECISION: Apply log transformation to target variable for modeling")

# =============================================================================
# 4. INDIVIDUAL FEATURE DISTRIBUTION ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("04. Individual feature distribution analysis")
print("=" * 60)

# Analyze numerical features distributions
print(f"\nAnalyzing distributions of {len(numerical_features)} numerical features...")

# Select top numerical features for detailed analysis
top_numerical = numerical_features[:12]  # Analyze first 12 numerical features

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for i in range(3):
    for j in range(4):
        idx = i * 4 + j
        if idx < len(top_numerical):
            feature = top_numerical[idx]
            values = X_df[feature].dropna()
            
            # Histogram
            axes[i, j].hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i, j].set_title(f'{feature}\nSkew: {stats.skew(values):.2f}')
            axes[i, j].set_xlabel(feature)
            axes[i, j].set_ylabel('Frequency')
            
            # Add statistics text
            mean_val = values.mean()
            median_val = values.median()
            axes[i, j].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
            axes[i, j].axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.1f}')
            axes[i, j].legend(fontsize=8)

plt.tight_layout()
plt.show()

# Analyze categorical features
print(f"\nAnalyzing distributions of {len(categorical_features)} categorical features...")

# Select top categorical features for detailed analysis
top_categorical = categorical_features[:8]  # Analyze first 8 categorical features

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i in range(2):
    for j in range(4):
        idx = i * 4 + j
        if idx < len(top_categorical):
            feature = top_categorical[idx]
            value_counts = X_df[feature].value_counts().head(10)  # Top 10 categories
            
            # Bar plot
            axes[i, j].bar(range(len(value_counts)), value_counts.values, color='lightcoral')
            axes[i, j].set_title(f'{feature}\n({len(value_counts)} categories shown)')
            axes[i, j].set_xlabel('Categories')
            axes[i, j].set_ylabel('Count')
            axes[i, j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# =============================================================================
# 5. RELATIONSHIP ANALYSIS WITH TARGET
# =============================================================================

print("\n" + "=" * 60)
print("05. Relationship analysis with target")
print("=" * 60)

# Select key features based on domain knowledge and availability
key_features = [
    'Gr Liv Area',      # Living area - most important for price
    'Total Bsmt SF',    # Basement area - important for value
    'Year Built',       # Age of house - affects value
    'Overall Qual',     # Overall quality - critical for price
    'Lot Area',         # Lot size - affects price
    'Kitchen Qual',     # Kitchen quality - important for buyers
    'Exter Qual',       # Exterior quality - first impression
    'Garage Cars',      # Garage capacity
    'Full Bath',        # Number of full bathrooms
    'Fireplaces'        # Number of fireplaces
]

# Verify feature availability
available_features = [f for f in key_features if f in X_df.columns]
print(f"\nAnalyzing relationships with target for: {available_features}")

# Create comprehensive relationship plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i in range(2):
    for j in range(3):
        idx = i * 3 + j
        if idx < len(available_features[:6]):
            feature = available_features[idx]
            
            if feature in numerical_features:
                # Numerical feature - scatter plot with trend line
                x = X_df[feature].values
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]
                
                if len(x_clean) > 0:
                    # Scatter plot
                    axes[i, j].scatter(x_clean, y_clean, alpha=0.6, s=20)
                    
                    # Add trend line
                    z = np.polyfit(x_clean, y_clean, 1)
                    p = np.poly1d(z)
                    axes[i, j].plot(x_clean, p(x_clean), "r--", alpha=0.8)
                    
                    # Calculate correlation
                    corr = np.corrcoef(x_clean, y_clean)[0, 1]
                    axes[i, j].set_title(f'{feature} vs SalePrice\nCorr: {corr:.3f}')
                    axes[i, j].set_xlabel(feature)
                    axes[i, j].set_ylabel('SalePrice ($)')
            
            else:
                # Categorical feature - box plot
                data_to_plot = []
                labels = []
                for category in X_df[feature].dropna().unique()[:6]:  # Top 6 categories
                    mask = X_df[feature] == category
                    data_to_plot.append(y[mask])
                    labels.append(str(category))
                
                if data_to_plot:
                    bp = axes[i, j].boxplot(data_to_plot, labels=labels)
                    axes[i, j].set_xlabel(feature)
                    axes[i, j].set_ylabel('SalePrice ($)')
                    axes[i, j].set_title(f'SalePrice by {feature}')
                    axes[i, j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# =============================================================================
# 6. CORRELATION ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("06. Correlation analysis")
print("=" * 60)

# Select numerical features for correlation analysis
numerical_for_corr = [f for f in available_features if f in numerical_features]

if len(numerical_for_corr) > 1:
    # Create correlation dataframe with features and target
    corr_df = X_df[numerical_for_corr].copy()
    corr_df['SalePrice'] = y
    corr_data = corr_df.corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Key Features')
    plt.tight_layout()
    plt.show()
    
    # Print top correlations with target
    target_correlations = corr_data['SalePrice'].sort_values(ascending=False)
    print("\nTop correlations with SalePrice:")
    print(target_correlations.head(10))

# =============================================================================
# 7. OUTLIER DETECTION AND ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("07. Outlier detection and analysis")
print("=" * 60)

print("Outlier detection using z-scores (threshold = 3):")

outlier_summary = {}
for feature in available_features[:8]:  # Analyze top 8 features
    if feature in numerical_features:
        x = X_df[feature].values
        mask = ~np.isnan(x)
        x_clean = x[mask]
        
        if len(x_clean) > 0:
            z_scores = np.abs((x_clean - x_clean.mean()) / x_clean.std())
            outliers_mask = z_scores > 3
            n_outliers = outliers_mask.sum()
            outlier_percentage = (n_outliers/len(x_clean)*100)
            
            outlier_summary[feature] = {
                'n_outliers': n_outliers,
                'percentage': outlier_percentage,
                'z_scores': z_scores,
                'outlier_mask': outliers_mask
            }
            
            print(f"{feature}: {n_outliers} outliers ({outlier_percentage:.1f}%)")

# Visualize outliers for top features
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        if idx < len(list(outlier_summary.keys())[:4]):
            feature = list(outlier_summary.keys())[idx]
            x = X_df[feature].values
            mask = ~np.isnan(x)
            x_clean = x[mask]
            y_clean = y[mask]
            
            outlier_mask = outlier_summary[feature]['outlier_mask']
            
            # Scatter plot with outliers highlighted
            axes[i, j].scatter(x_clean[~outlier_mask], y_clean[~outlier_mask], 
                             alpha=0.6, s=20, color='blue', label='Normal')
            axes[i, j].scatter(x_clean[outlier_mask], y_clean[outlier_mask], 
                             alpha=0.8, s=30, color='red', label='Outliers')
            
            axes[i, j].set_xlabel(feature)
            axes[i, j].set_ylabel('SalePrice ($)')
            axes[i, j].set_title(f'{feature} - Outliers Detection')
            axes[i, j].legend()

plt.tight_layout()
plt.show()

# =============================================================================
# 8. DATA QUALITY ISSUES AND INCONSISTENCIES
# =============================================================================

print("\n" + "=" * 60)
print("08. Data quality issues and inconsistencies")
print("=" * 60)

# Check for duplicate rows
duplicates = data_df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Check for inconsistent data types
print("\nChecking for potential data type inconsistencies:")
for feature in numerical_features:
    if X_df[feature].dtype == 'object':
        print(f"WARNING: {feature} is numerical but has object dtype")

# Check for extreme values that might be errors
print("\nChecking for potential data entry errors:")
for feature in available_features[:5]:
    if feature in numerical_features:
        values = X_df[feature].dropna()
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        extreme_low = (values < lower_bound).sum()
        extreme_high = (values > upper_bound).sum()
        
        if extreme_low > 0 or extreme_high > 0:
            print(f"{feature}: {extreme_low} extreme low, {extreme_high} extreme high values")

# =============================================================================
# 9. DATA CLEANING IMPLEMENTATION
# =============================================================================

print("\n" + "=" * 60)
print("09. Data cleaning implementation")
print("=" * 60)

# Create a copy for cleaning
X_clean = X_df.copy()
y_clean = y.copy()

print("IMPLEMENTING DATA CLEANING DECISIONS:")

# 1. Handle missing values
print("\n1. Handling missing values...")
for feature in features_with_missing.index:
    if feature in numerical_features:
        # Use median for numerical features
        median_val = X_clean[feature].median()
        X_clean[feature].fillna(median_val, inplace=True)
        print(f"   {feature}: Filled {features_with_missing[feature]} missing values with median ({median_val:.2f})")
    else:
        # Use mode for categorical features
        mode_val = X_clean[feature].mode()[0] if len(X_clean[feature].mode()) > 0 else 'Unknown'
        X_clean[feature].fillna(mode_val, inplace=True)
        print(f"   {feature}: Filled {features_with_missing[feature]} missing values with mode ('{mode_val}')")

# 2. Remove outliers using z-score method
print("\n2. Removing outliers...")
outliers_removed = 0
for feature in outlier_summary.keys():
    if outlier_summary[feature]['percentage'] > 5:  # Only remove if >5% outliers
        outlier_mask = outlier_summary[feature]['outlier_mask']
        feature_mask = ~np.isnan(X_df[feature].values)
        
        # Create global outlier mask
        global_outlier_mask = np.zeros(len(X_clean), dtype=bool)
        global_outlier_mask[feature_mask] = outlier_mask
        
        # Remove outliers
        X_clean = X_clean[~global_outlier_mask]
        y_clean = y_clean[~global_outlier_mask]
        
        outliers_removed += outlier_mask.sum()
        print(f"   {feature}: Removed {outlier_mask.sum()} outliers")

print(f"   Total outliers removed: {outliers_removed}")

# 3. Apply log transformation to target
print("\n3. Applying log transformation to target variable...")
y_clean_log = np.log(y_clean)
print(f"   Original target range: ${y_clean.min():,.0f} - ${y_clean.max():,.0f}")
print(f"   Log-transformed range: {y_clean_log.min():.2f} - {y_clean_log.max():.2f}")

# 4. Apply log transformation to skewed numerical features
print("\n4. Applying log transformation to skewed numerical features...")
skewed_features = []
for feature in numerical_features:
    if feature in X_clean.columns:
        values = X_clean[feature].dropna()
        if len(values) > 0 and values.min() > 0:  # Only for positive values
            skewness = stats.skew(values)
            if abs(skewness) > 1:  # Apply log if skewness > 1
                X_clean[f'{feature}_log'] = np.log(values)
                skewed_features.append(feature)
                print(f"   {feature}: Applied log transformation (skewness: {skewness:.2f})")

print(f"   Total features log-transformed: {len(skewed_features)}")

# 5. Create missing indicators for features with >10% missing values
print("\n5. Creating missing indicators...")
high_missing_features = features_with_missing[features_with_missing > len(X_df) * 0.1].index
for feature in high_missing_features:
    if feature in X_clean.columns:
        X_clean[f'{feature}_missing'] = X_df[feature].isnull().astype(int)
        print(f"   Created missing indicator for {feature}")

print(f"\nFinal cleaned dataset shape: {X_clean.shape}")
print(f"Final target shape: {y_clean_log.shape}")

# =============================================================================
# 10. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("10. Feature importance analysis")
print("=" * 60)

# Select cleaned numerical features for importance analysis
clean_numerical = [f for f in X_clean.columns if X_clean[f].dtype in ['int64', 'float64']]

if len(clean_numerical) > 1:
    # Calculate correlations with log-transformed target
    importance_df = X_clean[clean_numerical].copy()
    importance_df['SalePrice_log'] = y_clean_log
    correlations = importance_df.corr()['SalePrice_log'].abs().sort_values(ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = correlations.head(15)
    plt.barh(range(len(top_features)), top_features.values)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Absolute Correlation with log(SalePrice)')
    plt.title('Feature Importance (Correlation with Target)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 most important features:")
    print(correlations.head(10))

# =============================================================================
# 11. FINAL RECOMMENDATIONS FOR MODELING
# =============================================================================

print("\n" + "=" * 60)
print("11. Final recommendations for modeling")
print("=" * 60)

print("\n1. FEATURE SELECTION:")
print("   - Use log-transformed target variable (SalePrice_log)")
print("   - Focus on features with correlation > 0.3 with target")
print("   - Include missing indicators for high-missing features")
print("   - Consider polynomial features for non-linear relationships")

print("\n2. DATA PREPROCESSING:")
print("   - Missing values handled with median/mode imputation")
print("   - Outliers removed using z-score threshold of 3")
print("   - Log transformation applied to target and skewed features")
print("   - Missing indicators created for features with >10% missing values")

print("\n3. MODELING APPROACH:")
print("   - Start with simple linear regression on cleaned data")
print("   - Try polynomial regression for non-linear relationships")
print("   - Consider Huber loss for robustness to remaining outliers")
print("   - Use cross-validation for model selection")
print("   - Evaluate with RMSE and MAE metrics")

print("\n4. POTENTIAL ISSUES TO ADDRESS:")
print("   - Some features may need additional feature engineering")
print("   - Categorical features need encoding (one-hot or label encoding)")
print("   - Consider interaction terms between important features")
print("   - Scale features if using algorithms that require it")

print("\n" + "=" * 60)
print("EDA AND CLEANING COMPLETE - READY FOR MODELING")
print("=" * 60)
print(f"Original dataset: {data_df.shape}")
print(f"Cleaned dataset: {X_clean.shape}")
print(f"Features processed: {len(numerical_features) + len(categorical_features)}")
print(f"Missing values handled: {features_with_missing.sum()}")
print(f"Outliers removed: {outliers_removed}")
print("=" * 60)
