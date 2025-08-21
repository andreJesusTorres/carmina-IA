# =============================================================================
# EXPLORATORY DATA ANALYSIS AND DATA CLEANING
# Essential approach based on EPFL course standards
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. LOAD AND EXPLORE DATA STRUCTURE
# =============================================================================

# Load the dataset
data_df = pd.read_csv("house-prices.csv")
print(f"Dataset shape: {data_df.shape}")

# Extract target variable
y = data_df['SalePrice'].values
X_df = data_df.drop('SalePrice', axis=1)

print(f"Target variable range: ${y.min():,.0f} - ${y.max():,.0f}")

# =============================================================================
# 2. FEATURE EXPLORATION
# =============================================================================

# Identify data types
dtype_info = X_df.dtypes
numerical_features = dtype_info[dtype_info.isin(['int64', 'float64'])].index.tolist()
categorical_features = dtype_info[dtype_info == 'object'].index.tolist()

print(f"\nNumerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Check for missing values
missing_values = X_df.isnull().sum()
features_with_missing = missing_values[missing_values > 0]

if len(features_with_missing) > 0:
    print(f"\nFeatures with missing values: {len(features_with_missing)}")
    print("Top 5 features with most missing values:")
    print(features_with_missing.head().to_string())
else:
    print("\nNo missing values found")

# =============================================================================
# 3. TARGET VARIABLE ANALYSIS
# =============================================================================

print(f"\nTarget variable (SalePrice) analysis:")
print(f"Mean: ${y.mean():,.0f}, Median: ${np.median(y):,.0f}, Std: ${y.std():,.0f}")

# Visualize target distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(y, bins=30, alpha=0.7, color='blue')
plt.xlabel('SalePrice ($)')
plt.ylabel('Frequency')
plt.title('Distribution of SalePrice')

plt.subplot(1, 3, 2)
y_log = np.log(y)
plt.hist(y_log, bins=30, alpha=0.7, color='green')
plt.xlabel('log(SalePrice)')
plt.ylabel('Frequency')
plt.title('Distribution of log(SalePrice)')

plt.subplot(1, 3, 3)
plt.boxplot(y)
plt.ylabel('SalePrice ($)')
plt.title('Boxplot of SalePrice')

plt.tight_layout()
plt.show()

print("OBSERVATION: SalePrice distribution is right-skewed")
print("DECISION: Apply log transformation to make it more normal")

# =============================================================================
# 4. KEY FEATURES ANALYSIS
# =============================================================================

# Select key features based on domain knowledge
key_features = [
    'Gr Liv Area',      # Living area - most important for price
    'Total Bsmt SF',    # Basement area - important for value
    'Year Built',       # Age of house - affects value
    'Overall Qual',     # Overall quality - critical for price
    'Lot Area',         # Lot size - affects price
    'Kitchen Qual',     # Kitchen quality - important for buyers
    'Exter Qual'        # Exterior quality - first impression
]

# Verify feature availability
available_features = [f for f in key_features if f in X_df.columns]
print(f"\nAnalyzing key features: {available_features}")

# =============================================================================
# 5. RELATIONSHIP ANALYSIS WITH TARGET
# =============================================================================

plt.figure(figsize=(15, 10))

for i, feature in enumerate(available_features[:4]):  # Analyze top 4 features
    plt.subplot(2, 2, i+1)
    
    if feature in numerical_features:
        # Numerical feature - scatter plot
        x = X_df[feature].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) > 0:
            plt.scatter(x_clean, y_clean, alpha=0.6)
            plt.xlabel(feature)
            plt.ylabel('SalePrice ($)')
            plt.title(f'{feature} vs SalePrice')
    
    else:
        # Categorical feature - box plot
        data_to_plot = []
        labels = []
        for category in X_df[feature].dropna().unique()[:5]:  # Top 5 categories
            mask = X_df[feature] == category
            data_to_plot.append(y[mask])
            labels.append(str(category))
        
        if data_to_plot:
            plt.boxplot(data_to_plot, tick_labels=labels)
            plt.xlabel(feature)
            plt.ylabel('SalePrice ($)')
            plt.title(f'SalePrice by {feature}')
            plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# =============================================================================
# 6. OUTLIER DETECTION
# =============================================================================

print("\nOutlier detection using z-scores:")

for feature in available_features[:3]:  # Analyze top 3 numerical features
    if feature in numerical_features:
        x = X_df[feature].values
        mask = ~np.isnan(x)
        x_clean = x[mask]
        
        if len(x_clean) > 0:
            z_scores = np.abs((x_clean - x_clean.mean()) / x_clean.std())
            outliers_mask = z_scores > 2
            n_outliers = outliers_mask.sum()
            
            print(f"{feature}: {n_outliers} outliers ({(n_outliers/len(x_clean)*100):.1f}%)")

# =============================================================================
# 7. CORRELATION ANALYSIS
# =============================================================================

# Select numerical features for correlation analysis
numerical_for_corr = [f for f in available_features if f in numerical_features][:6]

if len(numerical_for_corr) > 1:
    # Create correlation dataframe with features and target
    corr_df = X_df[numerical_for_corr].copy()
    corr_df['SalePrice'] = y
    corr_data = corr_df.corr()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_data, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=45)
    plt.yticks(range(len(corr_data.columns)), corr_data.columns)
    plt.title('Correlation Matrix')
    
    # Add correlation values
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            plt.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                    ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 8. DATA CLEANING DECISIONS
# =============================================================================

print("\nDATA CLEANING DECISIONS:")

# Handle missing values
if len(features_with_missing) > 0:
    print("DECISION: Use median for numerical features, mode for categorical")
    print("DECISION: Create missing indicator for features with >10% missing values")
else:
    print("No missing values found in selected features")

print("DECISION: Remove outliers with |z-score| > 3 for numerical features")
print("DECISION: Apply log transformation to skewed numerical features")
print("DECISION: Apply log transformation to target variable (SalePrice)")

# =============================================================================
# 9. RECOMMENDATIONS FOR MODELING
# =============================================================================

print("\nRECOMMENDATIONS FOR MODELING:")

print("1. FEATURE SELECTION:")
print(f"   - Use numerical features: {[f for f in available_features if f in numerical_features]}")
print(f"   - Use categorical features: {[f for f in available_features if f in categorical_features]}")
print("   - Consider polynomial features for non-linear relationships")

print("\n2. DATA PREPROCESSING:")
print("   - Handle missing values using training data statistics only")
print("   - Remove outliers using z-score threshold of 3")
print("   - Apply log transformation to target variable")
print("   - Scale numerical features if using algorithms that require it")

print("\n3. MODELING APPROACH:")
print("   - Start with simple linear regression")
print("   - Try polynomial regression for non-linear relationships")
print("   - Consider Huber loss for robustness to outliers")
print("   - Use RSS and MAE for evaluation")

print("\n" + "=" * 60)
print("EDA AND CLEANING COMPLETE - READY FOR MODELING")
print("=" * 60)
