# =============================================================================
# HOUSE PRICES PREDICTION - PREPARACIÓN DE DATOS
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

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")

# Cargar los datos
train_data = pd.read_csv('house-prices.csv')
test_data = pd.read_csv('house-prices-test.csv')

print(f"Dataset de entrenamiento: {train_data.shape}")
print(f"Dataset de prueba: {test_data.shape}")

# Mostrar las primeras filas
print("\nPrimeras filas del dataset de entrenamiento:")
print(train_data.head())

print("\nColumnas disponibles:")
print(train_data.columns.tolist())

#Limpieza de datos -------------------------------------------------------------

# Separar la variable objetivo (SalePrice) del dataset de entrenamiento
X_train = train_data.drop(['SalePrice'], axis=1)
y_train = train_data['SalePrice']

print(f"Variables de entrada (X): {X_train.shape}")
print(f"Variable objetivo (y): {y_train.shape}")

# Verificar valores faltantes
print("\nValores faltantes en dataset de entrenamiento:")
missing_train = X_train.isnull().sum()
print(missing_train[missing_train > 0])

print("\nValores faltantes en dataset de prueba:")
missing_test = test_data.isnull().sum()
print(missing_test[missing_test > 0])

# Identificar columnas numéricas y categóricas
numeric_columns = X_train.select_dtypes(include=[np.number]).columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

print(f"\nColumnas numéricas: {len(numeric_columns)}")
print(f"Columnas categóricas: {len(categorical_columns)}")

# Rellenar valores faltantes
print("\nRellenando valores faltantes...")

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

print("✅ Valores faltantes rellenados")

# Verificar que no hay valores faltantes
print(f"\nValores faltantes restantes en X_train: {X_train.isnull().sum().sum()}")
print(f"Valores faltantes restantes en test_data: {test_data.isnull().sum().sum()}")

#Codificacion de variables categóricas ----------------------------------------

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
    print(f"✅ Codificada columna: {col}")

print(f"✅ Todas las {len(categorical_columns)} columnas categóricas han sido codificadas")

# Verificar que ahora todas las columnas son numéricas
print(f"\nTipos de datos finales:")
print(X_train.dtypes.value_counts())

# =============================================================================
# 4. ESCALADO DE VARIABLES
# =============================================================================

print("\n=== 4. ESCALADO DE VARIABLES ===")

# Crear escalador
scaler = StandardScaler()

# Ajustar el escalador con los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)
test_data_scaled = scaler.transform(test_data)

# Convertir de vuelta a DataFrame para mejor manejo
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.columns)

print("✅ Variables escaladas correctamente")
print(f"Forma final X_train: {X_train_scaled.shape}")
print(f"Forma final test_data: {test_data_scaled.shape}")

print("\n=== PREPARACIÓN DE DATOS COMPLETADA ===")
