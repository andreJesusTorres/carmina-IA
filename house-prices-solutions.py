# =============================================================================
# HOUSE PRICES PREDICTION - DATA SCIENCE PROJECT
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

# Cargar los datos
train_data = pd.read_csv('house-prices.csv')
test_data = pd.read_csv('house-prices-test.csv')

print(f"Dataset de entrenamiento: {train_data.shape}")
print(f"Dataset de prueba: {test_data.shape}")

# =============================================================================
# 2. PREPARACIÓN DE DATOS
# =============================================================================

# Separar la variable objetivo (SalePrice) del dataset de entrenamiento
X_train = train_data.drop(['SalePrice'], axis=1)
y_train = train_data['SalePrice']

# Identificar columnas numéricas y categóricas
numeric_columns = X_train.select_dtypes(include=[np.number]).columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

# Rellenar valores faltantes
# Para columnas numéricas: rellenar con la mediana
for col in numeric_columns:
    if X_train[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        test_data[col].fillna(median_val, inplace=True)

# Para columnas categóricas: rellenar con 'Unknown'
for col in categorical_columns:
    if X_train[col].isnull().sum() > 0:
        X_train[col].fillna('Unknown', inplace=True)
        test_data[col].fillna('Unknown', inplace=True)

# Verificar que no hay valores faltantes
print(f"\nValores faltantes restantes en X_train: {X_train.isnull().sum().sum()}")
print(f"Valores faltantes restantes en test_data: {test_data.isnull().sum().sum()}")

# =============================================================================
# 3. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# =============================================================================

# Crear codificadores para cada variable categórica
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    # Combinar datos de entrenamiento y prueba para obtener todas las categorías
    combined_data = pd.concat([X_train[col], test_data[col]]).unique()
    le.fit(combined_data)
    
    # Transformar datos de entrenamiento
    X_train[col] = le.transform(X_train[col])
    
    # Transformar datos de prueba
    test_data[col] = le.transform(test_data[col])
    
    label_encoders[col] = le

# =============================================================================
# 4. ESCALADO DE VARIABLES
# =============================================================================

# Crear escalador
scaler = StandardScaler()

# Ajustar el escalador con los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)
test_data_scaled = scaler.transform(test_data)

# Convertir de vuelta a DataFrame para mejor manejo
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.columns)

print("✅ Preparación de datos completada")
print(f"Forma final X_train: {X_train_scaled.shape}")
print(f"Forma final test_data: {test_data_scaled.shape}")

# =============================================================================
# 5. MODELADO
# =============================================================================

# Dividir datos de entrenamiento para validación
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Modelo 1: Regresión Lineal
print("\n=== Entrenando Regresión Lineal ===")
lr_model = LinearRegression()
lr_model.fit(X_train_split, y_train_split)
lr_pred = lr_model.predict(X_val)
lr_r2 = r2_score(y_val, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))
print(f"R² Score: {lr_r2:.4f}")
print(f"RMSE: {lr_rmse:.2f}")

# Modelo 2: Random Forest
print("\n=== Entrenando Random Forest ===")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)
rf_pred = rf_model.predict(X_val)
rf_r2 = r2_score(y_val, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
print(f"R² Score: {rf_r2:.4f}")
print(f"RMSE: {rf_rmse:.2f}")

# =============================================================================
# 6. PREDICCIONES FINALES
# =============================================================================

# Usar el mejor modelo para hacer predicciones en el test set
print("\n=== Generando predicciones finales ===")
if rf_r2 > lr_r2:
    final_predictions = rf_model.predict(test_data_scaled)
    print("Usando Random Forest para predicciones finales")
else:
    final_predictions = lr_model.predict(test_data_scaled)
    print("Usando Regresión Lineal para predicciones finales")

# Crear DataFrame con predicciones
predictions_df = pd.DataFrame({
    'Order': range(1, len(final_predictions) + 1),
    'Predicted_Price': final_predictions
})

# Guardar predicciones
predictions_df.to_csv('predictions.csv', index=False)
print("✅ Predicciones guardadas en 'predictions.csv'")

print("\n=== PROYECTO COMPLETADO ===")
