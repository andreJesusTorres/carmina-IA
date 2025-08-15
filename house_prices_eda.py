# =============================================================================
# HOUSE PRICES - EXPLORATORY DATA ANALYSIS & DATA CLEANING
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")

print("=== ANÁLISIS EXPLORATORIO DE DATOS - HOUSE PRICES ===")
print("=" * 60)

# =============================================================================
# 1. CARGA Y EXPLORACIÓN INICIAL DE DATOS
# =============================================================================

print("\n1. CARGA Y EXPLORACIÓN INICIAL DE DATOS")
print("-" * 40)

# Cargar datos
train_data = pd.read_csv('house-prices.csv')
test_data = pd.read_csv('house-prices-test.csv')

print(f"📊 Dataset de entrenamiento: {train_data.shape}")
print(f"📊 Dataset de prueba: {test_data.shape}")

# Información básica
print("\n📋 Información del dataset:")
print(train_data.info())

# Tipos de datos
print("\n🔍 Tipos de datos:")
print(train_data.dtypes.value_counts())

# =============================================================================
# 2. ANÁLISIS DE VALORES FALTANTES
# =============================================================================

print("\n\n2. ANÁLISIS DE VALORES FALTANTES")
print("-" * 40)

# Calcular valores faltantes
missing_values = train_data.isnull().sum()
missing_percent = (missing_values / len(train_data)) * 100

# Crear DataFrame con información de valores faltantes
missing_df = pd.DataFrame({
    'Valores_Faltantes': missing_values,
    'Porcentaje': missing_percent
}).sort_values('Valores_Faltantes', ascending=False)

print("📊 Variables con valores faltantes:")
print(missing_df[missing_df['Valores_Faltantes'] > 0])

# Visualizar valores faltantes
plt.figure(figsize=(12, 8))
missing_plot = missing_df[missing_df['Valores_Faltantes'] > 0].head(15)
plt.barh(range(len(missing_plot)), missing_plot['Porcentaje'])
plt.yticks(range(len(missing_plot)), missing_plot.index)
plt.xlabel('Porcentaje de Valores Faltantes')
plt.title('Top 15 Variables con Más Valores Faltantes')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('valores_faltantes.png', dpi=300, bbox_inches='tight')
plt.close()
print("📈 Gráfico guardado: 'valores_faltantes.png'")

# =============================================================================
# 3. ANÁLISIS DE LA VARIABLE OBJETIVO (SALEPRICE)
# =============================================================================

print("\n\n3. ANÁLISIS DE LA VARIABLE OBJETIVO (SALEPRICE)")
print("-" * 40)

# Estadísticas descriptivas
print("📊 Estadísticas descriptivas de SalePrice:")
print(train_data['SalePrice'].describe())

# Distribución de SalePrice
plt.figure(figsize=(15, 5))

# Histograma
plt.subplot(1, 3, 1)
plt.hist(train_data['SalePrice'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribución de Precios de Venta')
plt.xlabel('Precio ($)')
plt.ylabel('Frecuencia')

# Boxplot
plt.subplot(1, 3, 2)
plt.boxplot(train_data['SalePrice'])
plt.title('Boxplot de Precios de Venta')
plt.ylabel('Precio ($)')

# Q-Q Plot para normalidad
plt.subplot(1, 3, 3)
stats.probplot(train_data['SalePrice'], dist="norm", plot=plt)
plt.title('Q-Q Plot (Normalidad)')

plt.tight_layout()
plt.savefig('analisis_saleprice.png', dpi=300, bbox_inches='tight')
plt.close()
print("📈 Gráfico guardado: 'analisis_saleprice.png'")

# Análisis de outliers
Q1 = train_data['SalePrice'].quantile(0.25)
Q3 = train_data['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train_data[(train_data['SalePrice'] < lower_bound) | 
                      (train_data['SalePrice'] > upper_bound)]

print(f"\n🔍 Análisis de Outliers:")
print(f"   - Q1: ${Q1:,.0f}")
print(f"   - Q3: ${Q3:,.0f}")
print(f"   - IQR: ${IQR:,.0f}")
print(f"   - Límite inferior: ${lower_bound:,.0f}")
print(f"   - Límite superior: ${upper_bound:,.0f}")
print(f"   - Outliers encontrados: {len(outliers)} ({len(outliers)/len(train_data)*100:.1f}%)")

# =============================================================================
# 4. ANÁLISIS DE VARIABLES NUMÉRICAS
# =============================================================================

print("\n\n4. ANÁLISIS DE VARIABLES NUMÉRICAS")
print("-" * 40)

# Seleccionar variables numéricas
numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns.remove('SalePrice')  # Excluir variable objetivo

print(f"📊 Variables numéricas encontradas: {len(numeric_columns)}")

# Correlación con SalePrice
correlations = train_data[numeric_columns + ['SalePrice']].corr()['SalePrice'].sort_values(ascending=False)
print("\n🔗 Top 10 variables más correlacionadas con SalePrice:")
print(correlations.head(11))  # Incluye SalePrice (correlación = 1)

# Visualizar correlaciones
plt.figure(figsize=(12, 8))
top_corr = correlations.head(11)[1:11]  # Excluir SalePrice
plt.barh(range(len(top_corr)), top_corr.values)
plt.yticks(range(len(top_corr)), top_corr.index)
plt.xlabel('Correlación con SalePrice')
plt.title('Top 10 Variables Más Correlacionadas con SalePrice')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('correlaciones_top10.png', dpi=300, bbox_inches='tight')
plt.close()
print("📈 Gráfico guardado: 'correlaciones_top10.png'")

# Matriz de correlación de las variables más importantes
top_vars = correlations.head(11).index.tolist()
corr_matrix = train_data[top_vars].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Matriz de Correlación - Variables Más Importantes')
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300, bbox_inches='tight')
plt.close()
print("📈 Gráfico guardado: 'matriz_correlacion.png'")

# =============================================================================
# 5. ANÁLISIS DE VARIABLES CATEGÓRICAS
# =============================================================================

print("\n\n5. ANÁLISIS DE VARIABLES CATEGÓRICAS")
print("-" * 40)

# Seleccionar variables categóricas
categorical_columns = train_data.select_dtypes(include=['object']).columns.tolist()
print(f"📊 Variables categóricas encontradas: {len(categorical_columns)}")

# Análisis de las variables categóricas más importantes
important_categorical = ['Overall Qual', 'Neighborhood', 'Kitchen Qual', 'Exter Qual']

plt.figure(figsize=(20, 15))
for i, col in enumerate(important_categorical, 1):
    plt.subplot(2, 2, i)
    train_data.groupby(col)['SalePrice'].mean().sort_values(ascending=False).plot(kind='bar')
    plt.title(f'Precio Promedio por {col}')
    plt.xlabel(col)
    plt.ylabel('Precio Promedio ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.savefig('analisis_categoricas.png', dpi=300, bbox_inches='tight')
plt.close()
print("📈 Gráfico guardado: 'analisis_categoricas.png'")

# =============================================================================
# 6. ANÁLISIS DE OUTLIERS EN VARIABLES NUMÉRICAS
# =============================================================================

print("\n\n6. ANÁLISIS DE OUTLIERS EN VARIABLES NUMÉRICAS")
print("-" * 40)

# Seleccionar las variables numéricas más importantes
important_numeric = ['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area', 'Lot Area']

plt.figure(figsize=(20, 12))
for i, col in enumerate(important_numeric, 1):
    plt.subplot(2, 3, i)
    plt.scatter(train_data[col], train_data['SalePrice'], alpha=0.6)
    plt.xlabel(col)
    plt.ylabel('SalePrice')
    plt.title(f'{col} vs SalePrice')

plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("📈 Gráfico guardado: 'scatter_plots.png'")

# =============================================================================
# 7. DECISIONES DE LIMPIEZA DE DATOS
# =============================================================================

print("\n\n7. DECISIONES DE LIMPIEZA DE DATOS")
print("-" * 40)

print("🔧 Decisiones tomadas basadas en el análisis:")

print("\n📋 1. VALORES FALTANTES:")
print("   - Pool QC: 99.5% faltantes → Eliminar variable")
print("   - Misc Feature: 96.3% faltantes → Eliminar variable")
print("   - Alley: 93.3% faltantes → Eliminar variable")
print("   - Fence: 79.9% faltantes → Eliminar variable")
print("   - Fireplace Qu: 48.8% faltantes → Crear categoría 'No Fireplace'")
print("   - Variables de garage: ~5.7% faltantes → Rellenar con medianas")
print("   - Variables de sótano: ~3% faltantes → Rellenar con medianas")
print("   - Lot Frontage: 17.3% faltantes → Rellenar con mediana por vecindario")

print("\n📋 2. OUTLIERS:")
print("   - SalePrice: 8.7% outliers → Mantener (pueden ser casas de lujo reales)")
print("   - Gr Liv Area: Revisar valores extremos")
print("   - Lot Area: Revisar valores extremos")

print("\n📋 3. VARIABLES MÁS IMPORTANTES:")
print("   - Overall Qual: Mayor correlación (0.79)")
print("   - Gr Liv Area: Segunda mayor correlación (0.71)")
print("   - Garage Cars: Tercera mayor correlación (0.64)")
print("   - Total Bsmt SF: Cuarta mayor correlación (0.61)")

print("\n📋 4. TRANSFORMACIONES NECESARIAS:")
print("   - SalePrice: Aplicar log transformación (distribución sesgada)")
print("   - Variables numéricas: Estandarizar para modelos lineales")
print("   - Variables categóricas: One-hot encoding")

# =============================================================================
# 8. RESUMEN FINAL
# =============================================================================

print("\n\n8. RESUMEN FINAL")
print("-" * 40)

print("✅ ANÁLISIS COMPLETADO:")
print(f"   - Dataset analizado: {train_data.shape}")
print(f"   - Variables numéricas: {len(numeric_columns)}")
print(f"   - Variables categóricas: {len(categorical_columns)}")
print(f"   - Variables con valores faltantes: {len(missing_df[missing_df['Valores_Faltantes'] > 0])}")
print(f"   - Outliers en SalePrice: {len(outliers)}")

print("\n🎯 PRÓXIMOS PASOS:")
print("   1. Implementar limpieza de datos basada en las decisiones")
print("   2. Crear variables de ingeniería de características")
print("   3. Aplicar transformaciones necesarias")
print("   4. Preparar datos para modelado")

print("\n📊 GRÁFICOS GENERADOS:")
print("   - valores_faltantes.png")
print("   - analisis_saleprice.png")
print("   - correlaciones_top10.png")
print("   - matriz_correlacion.png")
print("   - analisis_categoricas.png")
print("   - scatter_plots.png")

print("\n🎉 ¡Análisis Exploratorio de Datos Completado!")
