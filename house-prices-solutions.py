# =============================================================================
# HOUSE PRICES PREDICTION - JUNIOR DATA SCIENCE PROJECT
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
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
print("Generando gráficos de distribución...")
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
# Guardar el gráfico en lugar de mostrarlo
plt.savefig('distribucion_precios.png')
plt.close()
print("Gráfico guardado como 'distribucion_precios.png'")

# =============================================================================
# 3. PREPARACIÓN DE DATOS
# =============================================================================

print("\n=== PREPARACIÓN DE DATOS ===")

# Separar variable objetivo
X_train = train_data.drop(['SalePrice'], axis=1)
y_train = train_data['SalePrice']

# Manejo simple de valores faltantes
print("Rellenando valores faltantes...")

numeric_columns = X_train.select_dtypes(include=[np.number]).columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

X_train[numeric_columns] = X_train[numeric_columns].fillna(X_train[numeric_columns].median())
test_data[numeric_columns] = test_data[numeric_columns].fillna(test_data[numeric_columns].median())

X_train[categorical_columns] = X_train[categorical_columns].fillna('Unknown')
test_data[categorical_columns] = test_data[categorical_columns].fillna('Unknown')

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

# Modelo 3: Huber Regressor
print("Entrenando Huber Regressor...")
huber_model = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001)
huber_model.fit(X_train_split, y_train_split)
huber_pred = huber_model.predict(X_val)
huber_r2 = r2_score(y_val, huber_pred)
huber_rmse = np.sqrt(mean_squared_error(y_val, huber_pred))
print(f"Huber Regressor - R²: {huber_r2:.4f}, RMSE: {huber_rmse:.2f}")

# =============================================================================
# 5. PREDICCIONES FINALES
# =============================================================================

print("\n=== PREDICCIONES FINALES ===")

# Usar Random Forest para las predicciones (modelo más simple y efectivo)
print("🌲 Haciendo predicciones con Random Forest...")
final_predictions = rf_model.predict(test_data)

# Crear un DataFrame con las predicciones
predictions_df = pd.DataFrame({
    'ID_Casa': range(1, len(final_predictions) + 1),
    'Precio_Predicho': final_predictions
})

# Mostrar las primeras 10 predicciones
print(f"\n📋 Primeras 10 predicciones:")
print(predictions_df.head(10))

# Mostrar estadísticas de las predicciones
print(f"\n📊 Estadísticas de las predicciones:")
print(f"💰 Precio mínimo predicho: ${predictions_df['Precio_Predicho'].min():,.0f}")
print(f"💰 Precio máximo predicho: ${predictions_df['Precio_Predicho'].max():,.0f}")
print(f"💰 Precio promedio predicho: ${predictions_df['Precio_Predicho'].mean():,.0f}")
