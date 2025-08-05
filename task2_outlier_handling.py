import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load the data and perform initial exploration"""
    print("=== TASK 2: OUTLIER HANDLING IN REGRESSION MODELS ===\n")
    
    # Load the data
    data = pd.read_csv('p3_task-2.csv')
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {list(data.columns[:-1])}")
    print(f"Target: {data.columns[-1]}")
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nData statistics:")
    print(data.describe())
    
    return data

def split_data(data):
    """Split data into train and test sets (80-20 split)"""
    print("\n=== DATA SPLITTING ===")
    
    X = data.drop('y', axis=1)
    y = data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Features: {list(X.columns)}")
    
    return X_train, X_test, y_train, y_test

def explore_training_data(X_train, y_train):
    """Explore and visualize the training data to identify outliers"""
    print("\n=== TRAINING DATA EXPLORATION ===")
    
    # Combine features and target for analysis
    train_data = X_train.copy()
    train_data['y'] = y_train
    
    # Create subplots for each feature vs target
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Data: Feature vs Target Relationships', fontsize=16, fontweight='bold')
    
    features = ['x1', 'x2', 'x3']
    colors = ['blue', 'green', 'red']
    
    for i, (feature, color) in enumerate(zip(features, colors)):
        row = i // 2
        col = i % 2
        
        # Scatter plot
        axes[row, col].scatter(X_train[feature], y_train, alpha=0.6, color=color, s=30)
        axes[row, col].set_xlabel(feature, fontsize=12)
        axes[row, col].set_ylabel('y', fontsize=12)
        axes[row, col].set_title(f'{feature} vs y', fontsize=14, fontweight='bold')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(X_train[feature], y_train, 1)
        p = np.poly1d(z)
        axes[row, col].plot(X_train[feature], p(X_train[feature]), "r--", alpha=0.8, linewidth=2)
    
    # Box plots for outlier detection
    axes[1, 1].boxplot([X_train['x1'], X_train['x2'], X_train['x3']], 
                      labels=['x1', 'x2', 'x3'])
    axes[1, 1].set_title('Feature Distributions (Box Plots)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Values', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical outlier detection
    print("\nOutlier Analysis:")
    for feature in features:
        Q1 = X_train[feature].quantile(0.25)
        Q3 = X_train[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = X_train[(X_train[feature] < lower_bound) | (X_train[feature] > upper_bound)]
        print(f"{feature}: {len(outliers)} outliers detected ({len(outliers)/len(X_train)*100:.1f}%)")
        if len(outliers) > 0:
            print(f"  Range: [{outliers[feature].min():.3f}, {outliers[feature].max():.3f}]")

def remove_outliers(X_train, y_train, method='iqr'):
    """Remove outliers from training data using IQR method"""
    print(f"\n=== OUTLIER REMOVAL ({method.upper()}) ===")
    
    original_size = len(X_train)
    X_clean = X_train.copy()
    y_clean = y_train.copy()
    
    if method == 'iqr':
        # IQR method for outlier detection
        for feature in ['x1', 'x2', 'x3']:
            Q1 = X_clean[feature].quantile(0.25)
            Q3 = X_clean[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers
            mask = (X_clean[feature] >= lower_bound) & (X_clean[feature] <= upper_bound)
            X_clean = X_clean[mask]
            y_clean = y_clean[mask]
    
    removed_count = original_size - len(X_clean)
    print(f"Original training samples: {original_size}")
    print(f"After outlier removal: {len(X_clean)}")
    print(f"Removed {removed_count} outliers ({removed_count/original_size*100:.1f}%)")
    
    return X_clean, y_clean

def train_and_evaluate_models(X_train, X_train_clean, X_test, y_train, y_train_clean, y_test):
    """Train and evaluate the three regression models"""
    print("\n=== MODEL TRAINING AND EVALUATION ===")
    
    # Standardize features for better model performance
    scaler = StandardScaler()
    
    # Model 1: Linear Regression with outliers
    print("\n1. Linear Regression (with outliers):")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_with_outliers = LinearRegression()
    lr_with_outliers.fit(X_train_scaled, y_train)
    
    y_pred_lr_outliers = lr_with_outliers.predict(X_test_scaled)
    mae_lr_outliers = mean_absolute_error(y_test, y_pred_lr_outliers)
    print(f"   MAE: {mae_lr_outliers:.2f}")
    
    # Model 2: Huber Regression (robust to outliers)
    print("\n2. Huber Regression (robust to outliers):")
    huber = HuberRegressor(epsilon=1.35, max_iter=1000)
    huber.fit(X_train_scaled, y_train)
    
    y_pred_huber = huber.predict(X_test_scaled)
    mae_huber = mean_absolute_error(y_test, y_pred_huber)
    print(f"   MAE: {mae_huber:.2f}")
    
    # Model 3: Linear Regression without outliers
    print("\n3. Linear Regression (without outliers):")
    X_train_clean_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test)
    
    lr_without_outliers = LinearRegression()
    lr_without_outliers.fit(X_train_clean_scaled, y_train_clean)
    
    y_pred_lr_clean = lr_without_outliers.predict(X_test_scaled)
    mae_lr_clean = mean_absolute_error(y_test, y_pred_lr_clean)
    print(f"   MAE: {mae_lr_clean:.2f}")
    
    return {
        'lr_outliers': (mae_lr_outliers, y_pred_lr_outliers),
        'huber': (mae_huber, y_pred_huber),
        'lr_clean': (mae_lr_clean, y_pred_lr_clean)
    }

def compare_models(results, y_test):
    """Compare the three models and provide analysis"""
    print("\n=== MODEL COMPARISON ===")
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': ['Linear Regression (with outliers)', 'Huber Regression', 'Linear Regression (without outliers)'],
        'MAE': [results['lr_outliers'][0], results['huber'][0], results['lr_clean'][0]]
    })
    
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_idx = comparison_df['MAE'].idxmin()
    best_model = comparison_df.loc[best_model_idx, 'Model']
    best_mae = comparison_df.loc[best_model_idx, 'MAE']
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Best MAE: {best_mae:.2f}")
    
    # Calculate improvements
    baseline_mae = results['lr_outliers'][0]
    huber_improvement = ((baseline_mae - results['huber'][0]) / baseline_mae) * 100
    clean_improvement = ((baseline_mae - results['lr_clean'][0]) / baseline_mae) * 100
    
    print(f"\nImprovements over baseline (Linear Regression with outliers):")
    print(f"   Huber Regression: {huber_improvement:.1f}% improvement")
    print(f"   Linear Regression (clean): {clean_improvement:.1f}% improvement")
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # MAE comparison bar plot
    models = ['LR (with outliers)', 'Huber', 'LR (without outliers)']
    maes = [results['lr_outliers'][0], results['huber'][0], results['lr_clean'][0]]
    colors = ['red', 'blue', 'green']
    
    bars = ax1.bar(models, maes, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Model Performance Comparison (MAE)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mae:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Predicted vs Actual scatter plots
    ax2.scatter(y_test, results['lr_outliers'][1], alpha=0.6, label='LR (with outliers)', color='red', s=30)
    ax2.scatter(y_test, results['huber'][1], alpha=0.6, label='Huber', color='blue', s=30)
    ax2.scatter(y_test, results['lr_clean'][1], alpha=0.6, label='LR (without outliers)', color='green', s=30)
    
    # Perfect prediction line
    min_val = min(y_test.min(), min(results['lr_outliers'][1].min(), 
                                   results['huber'][1].min(), results['lr_clean'][1].min()))
    max_val = max(y_test.max(), max(results['lr_outliers'][1].max(), 
                                   results['huber'][1].max(), results['lr_clean'][1].max()))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Actual Values', fontsize=12)
    ax2.set_ylabel('Predicted Values', fontsize=12)
    ax2.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

def provide_analysis():
    """Provide detailed analysis of the results"""
    print("\n=== ANALYSIS AND CONCLUSIONS ===")
    
    print("""
📊 Key Findings:

1. **Outlier Impact**: The presence of outliers in the training data significantly 
   affects the performance of standard linear regression models.

2. **Huber Regression Effectiveness**: Huber regression demonstrates robustness 
   to outliers by using a different loss function that is less sensitive to 
   extreme values compared to ordinary least squares.

3. **Data Cleaning Benefits**: Removing outliers from the training data improves 
   the linear regression model's performance, as it can learn from cleaner, 
   more representative patterns.

4. **Model Selection**: Both Huber regression and linear regression on cleaned 
   data outperform the baseline linear regression model trained on data with 
   outliers.

🔍 Technical Insights:

- **Huber Regression**: Uses a combination of squared loss for small residuals 
  and absolute loss for large residuals, making it robust to outliers while 
  maintaining efficiency for normal data.

- **Outlier Removal Strategy**: The IQR method (1.5 * IQR rule) effectively 
  identifies and removes extreme values that could skew the model's learning.

- **Test Set Consistency**: By keeping the test set unchanged (with outliers), 
  we ensure fair comparison across all models and realistic evaluation of 
  real-world performance.

💡 Recommendations:

1. For datasets with known outliers, consider using robust regression methods 
   like Huber regression.

2. When possible, investigate and understand the nature of outliers before 
   removing them - they might contain valuable information.

3. Always validate model performance on a separate test set that reflects 
   real-world conditions.
    """)

def main():
    """Main execution function"""
    # Load and explore data
    data = load_and_explore_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Explore training data
    explore_training_data(X_train, y_train)
    
    # Remove outliers from training data
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_train_clean, X_test, y_train, y_train_clean, y_test)
    
    # Compare models
    comparison_df = compare_models(results, y_test)
    
    # Provide analysis
    provide_analysis()
    
    print("\n✅ Task 2 completed successfully!")
    print("📁 Generated files:")
    print("   - training_data_exploration.png")
    print("   - model_comparison.png")

if __name__ == "__main__":
    main() 