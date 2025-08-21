# =============================================================================
# ANÁLISIS EXPLORATORIO DE DATOS (EDA) Y LIMPIEZA DE DATOS
# HOUSE PRICES PREDICTION - AMES IOWA DATASET
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# 1. CARGA Y EXPLORACIÓN INICIAL
# =============================================================================

print("=== CARGA Y EXPLORACIÓN INICIAL ===")

# Cargar datasets
train_data = pd.read_csv('house-prices.csv')
test_data = pd.read_csv('house-prices-test.csv')

# Información básica
print("Forma de los datasets:")
print(f"Train: {train_data.shape}")
print(f"Test: {test_data.shape}")

# Información de columnas
print("\nInformación de columnas:")
print(train_data.info())

# Primeras filas
print("\nPrimeras filas:")
print(train_data.head())

# =============================================================================
# 2. ANÁLISIS DE VALORES FALTANTES
# =============================================================================

print("\n=== ANÁLISIS DE VALORES FALTANTES ===")

# Identificar valores faltantes
missing_values = train_data.isnull().sum()
missing_percentage = (missing_values / len(train_data)) * 100

# Crear DataFrame con información de valores faltantes
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing_Count': missing_values.values,
    'Missing_Percentage': missing_percentage.values
})

# Filtrar solo columnas con valores faltantes
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

print("Valores faltantes por columna:")
print(missing_df)

# Visualizar valores faltantes
plt.figure(figsize=(15, 8))
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Mapa de Valores Faltantes', fontsize=16)
plt.tight_layout()
plt.show()

# =============================================================================
# 3. ANÁLISIS DE LA VARIABLE OBJETIVO (SalePrice)
# =============================================================================

print("\n=== ANÁLISIS DE LA VARIABLE OBJETIVO (SalePrice) ===")

# Estadísticas descriptivas de SalePrice
print("Estadísticas de SalePrice:")
print(train_data['SalePrice'].describe())

# Distribución de SalePrice
plt.figure(figsize=(15, 5))

# Histograma
plt.subplot(1, 3, 1)
plt.hist(train_data['SalePrice'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribución de Precios de Venta', fontsize=12)
plt.xlabel('Precio de Venta')
plt.ylabel('Frecuencia')

# Boxplot
plt.subplot(1, 3, 2)
plt.boxplot(train_data['SalePrice'])
plt.title('Boxplot de Precios de Venta', fontsize=12)
plt.ylabel('Precio de Venta')

# Q-Q plot para normalidad
plt.subplot(1, 3, 3)
stats.probplot(train_data['SalePrice'], dist="norm", plot=plt)
plt.title('Q-Q Plot para Normalidad', fontsize=12)

plt.tight_layout()
plt.show()

# Identificar outliers en SalePrice
Q1 = train_data['SalePrice'].quantile(0.25)
Q3 = train_data['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train_data[(train_data['SalePrice'] < lower_bound) | (train_data['SalePrice'] > upper_bound)]
print(f"\nOutliers en SalePrice: {len(outliers)} observaciones")
print(f"Límite inferior: ${lower_bound:,.0f}")
print(f"Límite superior: ${upper_bound:,.0f}")

# =============================================================================
# 4. ANÁLISIS DE VARIABLES NUMÉRICAS
# =============================================================================

print("\n=== ANÁLISIS DE VARIABLES NUMÉRICAS ===")

# Seleccionar variables numéricas
numeric_columns = train_data.select_dtypes(include=[np.number]).columns
numeric_data = train_data[numeric_columns]

# Estadísticas descriptivas
print("Estadísticas descriptivas de variables numéricas:")
print(numeric_data.describe())

# Matriz de correlación
plt.figure(figsize=(20, 16))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación de Variables Numéricas', fontsize=16)
plt.tight_layout()
plt.show()

# Correlación con SalePrice
correlation_with_target = correlation_matrix['SalePrice'].sort_values(ascending=False)
print("\nCorrelación con SalePrice:")
print(correlation_with_target)

# Top 10 variables más correlacionadas con SalePrice
top_correlations = correlation_with_target.head(11)  # Excluyendo SalePrice
print("\nTop 10 variables más correlacionadas con SalePrice:")
print(top_correlations)

# =============================================================================
# 5. ANÁLISIS DE VARIABLES CATEGÓRICAS
# =============================================================================

print("\n=== ANÁLISIS DE VARIABLES CATEGÓRICAS ===")

# Seleccionar variables categóricas
categorical_columns = train_data.select_dtypes(include=['object']).columns
categorical_data = train_data[categorical_columns]

print(f"Número de variables categóricas: {len(categorical_columns)}")

# Análisis de variables categóricas más importantes
important_categorical = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', 'Garage Area']

for col in important_categorical:
    if col in categorical_columns:
        print(f"\n=== Análisis de {col} ===")
        print(f"Valores únicos: {train_data[col].nunique()}")
        print(f"Valores más frecuentes:")
        print(train_data[col].value_counts().head())
        
        # Relación con SalePrice
        plt.figure(figsize=(12, 5))
        
        # Boxplot
        plt.subplot(1, 2, 1)
        train_data.boxplot(column='SalePrice', by=col, ax=plt.gca())
        plt.title(f'Precio de Venta por {col}', fontsize=12)
        plt.suptitle('')  # Eliminar título automático
        
        # Gráfico de barras con precio promedio
        plt.subplot(1, 2, 2)
        avg_price = train_data.groupby(col)['SalePrice'].mean().sort_values(ascending=False)
        avg_price.plot(kind='bar')
        plt.title(f'Precio Promedio por {col}', fontsize=12)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# 6. ANÁLISIS DE OUTLIERS
# =============================================================================

print("\n=== ANÁLISIS DE OUTLIERS ===")

# Identificar outliers en variables numéricas importantes
important_numeric = ['Lot Area', 'Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Garage Area']

plt.figure(figsize=(20, 12))
for i, col in enumerate(important_numeric, 1):
    plt.subplot(2, 3, i)
    plt.boxplot(train_data[col].dropna())
    plt.title(f'Boxplot de {col}', fontsize=12)
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# Detectar outliers usando IQR
outliers_summary = {}
for col in important_numeric:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_count = len(train_data[(train_data[col] < lower_bound) | (train_data[col] > upper_bound)])
    outliers_summary[col] = outliers_count

print("Resumen de outliers por variable:")
for col, count in outliers_summary.items():
    print(f"{col}: {count} outliers")

# =============================================================================
# 7. ANÁLISIS DE RELACIONES ESPECÍFICAS
# =============================================================================

print("\n=== ANÁLISIS DE RELACIONES ESPECÍFICAS ===")

# Relación entre área de vivienda y precio
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(train_data['Gr Liv Area'], train_data['SalePrice'], alpha=0.6)
plt.xlabel('Área de Vivienda')
plt.ylabel('Precio de Venta')
plt.title('Precio vs Área de Vivienda', fontsize=12)

plt.subplot(1, 3, 2)
plt.scatter(train_data['Lot Area'], train_data['SalePrice'], alpha=0.6)
plt.xlabel('Área del Lote')
plt.ylabel('Precio de Venta')
plt.title('Precio vs Área del Lote', fontsize=12)

plt.subplot(1, 3, 3)
plt.scatter(train_data['Total Bsmt SF'], train_data['SalePrice'], alpha=0.6)
plt.xlabel('Área Total del Sótano')
plt.ylabel('Precio de Venta')
plt.title('Precio vs Área del Sótano', fontsize=12)

plt.tight_layout()
plt.show()

# Relación entre calidad general y precio
plt.figure(figsize=(10, 6))
train_data.boxplot(column='SalePrice', by='Overall Qual')
plt.title('Precio de Venta por Calidad General', fontsize=12)
plt.suptitle('')
plt.show()

# =============================================================================
# 8. LIMPIEZA DE DATOS
# =============================================================================

print("\n=== LIMPIEZA DE DATOS ===")

# 8.1 Manejo de Valores Faltantes
def handle_missing_values(df):
    """
    Maneja los valores faltantes en el dataset
    - Variables numéricas: rellenar con mediana
    - Variables categóricas: rellenar con 'Unknown'
    """
    df_clean = df.copy()
    
    # Variables numéricas - rellenar con mediana
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Variables categóricas - rellenar con 'Unknown'
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna('Unknown', inplace=True)
    
    return df_clean

# Aplicar limpieza
print("Aplicando limpieza de valores faltantes...")
train_clean = handle_missing_values(train_data)
test_clean = handle_missing_values(test_data)

print("Valores faltantes después de la limpieza:")
print(f"Train: {train_clean.isnull().sum().sum()}")
print(f"Test: {test_clean.isnull().sum().sum()}")

# 8.2 Manejo de Outliers (opcional)
def remove_outliers(df, column, n_std=3):
    """
    Elimina outliers basándose en desviaciones estándar
    """
    df_clean = df.copy()
    mean = df_clean[column].mean()
    std = df_clean[column].std()
    
    # Eliminar valores más allá de n_std desviaciones estándar
    df_clean = df_clean[(df_clean[column] <= mean + n_std * std) & 
                       (df_clean[column] >= mean - n_std * std)]
    
    return df_clean

# Comentado por defecto - descomenta si quieres eliminar outliers
# print("Eliminando outliers extremos...")
# train_clean = remove_outliers(train_clean, 'SalePrice', n_std=3)

# 8.3 Codificación de Variables Categóricas
def encode_categorical(df):
    """
    Codifica variables categóricas usando one-hot encoding
    """
    df_encoded = df.copy()
    
    # One-hot encoding para variables categóricas
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    
    # Aplicar one-hot encoding
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=True)
    
    return df_encoded

# Aplicar codificación
print("Aplicando codificación de variables categóricas...")
train_encoded = encode_categorical(train_clean)
test_encoded = encode_categorical(test_clean)

print(f"Forma después de codificación - Train: {train_encoded.shape}")
print(f"Forma después de codificación - Test: {test_encoded.shape}")

# =============================================================================
# 9. ANÁLISIS FINAL Y CONCLUSIONES
# =============================================================================

print("\n=== ANÁLISIS FINAL Y CONCLUSIONES ===")

# Resumen de la limpieza
print("=== RESUMEN DE LA LIMPIEZA DE DATOS ===")
print(f"Observaciones originales: {len(train_data)}")
print(f"Observaciones después de limpieza: {len(train_clean)}")
print(f"Variables originales: {len(train_data.columns)}")
print(f"Variables después de codificación: {len(train_encoded.columns)}")

# Estadísticas finales del dataset limpio
print("\n=== ESTADÍSTICAS FINALES ===")
print("Estadísticas de SalePrice después de limpieza:")
print(train_clean['SalePrice'].describe())

# Guardar datos limpios
print("\nGuardando datos limpios...")
train_encoded.to_csv('house_prices_cleaned.csv', index=False)
test_encoded.to_csv('house_prices_test_cleaned.csv', index=False)

print("Datos limpios guardados exitosamente!")

# =============================================================================
# 10. INSIGHTS Y RECOMENDACIONES
# =============================================================================

print("\n=== INSIGHTS Y RECOMENDACIONES ===")

print("""
INSIGHTS PRINCIPALES:

1. VARIABLES MÁS IMPORTANTES PARA PREDECIR EL PRECIO:
   - Overall Qual (Calidad General): Correlación muy alta con el precio
   - Gr Liv Area (Área de Vivienda): Correlación positiva fuerte
   - Garage Cars (Capacidad del Garaje): Importante para el valor
   - Total Bsmt SF (Área Total del Sótano): Añade valor significativo
   - 1st Flr SF (Área del Primer Piso): Correlación positiva

2. PATRONES IDENTIFICADOS:
   - El precio de venta tiene una distribución asimétrica (sesgada a la derecha)
   - Existen outliers en el extremo superior de precios
   - La calidad general es el factor más determinante del precio
   - Las casas más grandes tienden a ser más caras, pero no linealmente

3. DECISIONES DE LIMPIEZA TOMADAS:
   - Valores faltantes numéricos: Rellenados con mediana
   - Valores faltantes categóricos: Rellenados con 'Unknown'
   - Variables categóricas: Codificadas con one-hot encoding
   - Outliers: Mantenidos para preservar información valiosa

4. RECOMENDACIONES PARA EL MODELADO:
   - Usar transformaciones logarítmicas para el precio de venta
   - Considerar feature engineering para crear nuevas variables
   - Implementar validación cruzada robusta
   - Evaluar múltiples algoritmos (Random Forest, XGBoost, etc.)
   - Considerar ensemble methods para mejorar la precisión
""")

# =============================================================================
# 11. FUNCIONES AUXILIARES PARA FUTURO USO
# =============================================================================

def create_feature_importance_plot(model, feature_names, top_n=20):
    """
    Crea un gráfico de importancia de características
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top {top_n} Características Más Importantes')
        plt.bar(range(top_n), importances[indices[:top_n]])
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

def plot_residuals(y_true, y_pred):
    """
    Crea gráficos de residuos para evaluar el modelo
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(15, 5))
    
    # Gráfico de residuos vs predicciones
    plt.subplot(1, 3, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.title('Residuos vs Predicciones')
    
    # Histograma de residuos
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residuos')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Residuos')
    
    # Q-Q plot de residuos
    plt.subplot(1, 3, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot de Residuos')
    
    plt.tight_layout()
    plt.show()

print("\n=== ANÁLISIS COMPLETADO ===")
print("El archivo contiene:")
print("1. Análisis exploratorio completo de los datos")
print("2. Identificación y manejo de valores faltantes")
print("3. Análisis de outliers y relaciones entre variables")
print("4. Limpieza y preparación de datos")
print("5. Funciones auxiliares para el modelado")
print("6. Insights y recomendaciones para el siguiente paso")
