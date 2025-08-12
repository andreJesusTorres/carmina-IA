# =============================================================================
# HOUSE PRICES PREDICTION - JUNIOR DATA SCIENCE PROJECT
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

print("=== CARGA DE DATOS ===")
train_data = pd.read_csv('house-prices.csv')
test_data = pd.read_csv('house-prices-test.csv')

print(f"Dataset de entrenamiento: {train_data.shape}")
print(f"Dataset de prueba: {test_data.shape}")
print(f"Columnas en train: {train_data.columns.tolist()[:5]}...")  # Mostrar primeras 5 columnas

# =============================================================================
# 2. EXPLORACIÓN DE DATOS (EDA)
# =============================================================================

print("\n=== EXPLORACIÓN DE DATOS ===")

# Información básica del dataset
print("Información del dataset:")
print(train_data.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(train_data.describe())

# Verificar valores faltantes
print("\nValores faltantes:")
missing_values = train_data.isnull().sum()
print(missing_values[missing_values > 0])

# Distribución de la variable objetivo
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(train_data['SalePrice'], bins=30, alpha=0.7)
plt.title('Distribución de Precios de Venta')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
plt.boxplot(train_data['SalePrice'])
plt.title('Boxplot de Precios de Venta')
plt.ylabel('Precio')

plt.tight_layout()
plt.show()

# =============================================================================
# 3. PREPARACIÓN DE DATOS
# =============================================================================

print("\n=== PREPARACIÓN DE DATOS ===")

# Separar variable objetivo
X_train = train_data.drop(['SalePrice'], axis=1)
y_train = train_data['SalePrice']

# Manejo simple de valores faltantes
print("Rellenando valores faltantes...")
X_train = X_train.fillna(X_train.median())  # Para numéricas
X_train = X_train.fillna('Unknown')         # Para categóricas
test_data = test_data.fillna(test_data.median())
test_data = test_data.fillna('Unknown')

# Convertir variables categóricas a numéricas (método simple)
print("Convirtiendo variables categóricas...")
X_train = pd.get_dummies(X_train)
test_data = pd.get_dummies(test_data)

# Alinear columnas entre train y test
common_columns = X_train.columns.intersection(test_data.columns)
X_train = X_train[common_columns]
test_data = test_data[common_columns]

print(f"Forma final X_train: {X_train.shape}")
print(f"Forma final test_data: {test_data.shape}")

# =============================================================================
# 4. MODELADO
# =============================================================================

print("\n=== MODELADO ===")

# Dividir datos para validación
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Modelo 1: Regresión Lineal
print("Entrenando Regresión Lineal...")
lr_model = LinearRegression()
lr_model.fit(X_train_split, y_train_split)
lr_pred = lr_model.predict(X_val)
lr_r2 = r2_score(y_val, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))
print(f"Regresión Lineal - R²: {lr_r2:.4f}, RMSE: {lr_rmse:.2f}")

# Modelo 2: Random Forest
print("Entrenando Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)
rf_pred = rf_model.predict(X_val)
rf_r2 = r2_score(y_val, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
print(f"Random Forest - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.2f}")

# =============================================================================
# 5. PREDICCIONES FINALES
# =============================================================================

print("\n=== PREDICCIONES FINALES ===")

# Usar el mejor modelo
if rf_r2 > lr_r2:
    final_predictions = rf_model.predict(test_data)
    print("Usando Random Forest (mejor modelo)")
else:
    final_predictions = lr_model.predict(test_data)
    print("Usando Regresión Lineal (mejor modelo)")

# Guardar predicciones
predictions_df = pd.DataFrame({
    'Order': range(1, len(final_predictions) + 1),
    'Predicted_Price': final_predictions
})

predictions_df.to_csv('predictions.csv', index=False)
print("✅ Predicciones guardadas en 'predictions.csv'")

print("\n=== PROYECTO COMPLETADO ===")
print(f"Se predijeron precios para {len(final_predictions)} casas")
