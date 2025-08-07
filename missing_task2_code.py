# ============================================================================
# MODELO 1: Linear Regression CON outliers (FALTA ESTO)
# ============================================================================
print("\n" + "="*60)
print("MODELO 1: Linear Regression CON outliers")
print("="*60)

# Entrenar modelo con datos originales (con outliers)
linear_with_outliers = LinearRegression()
linear_with_outliers.fit(x_train, y_train)

# Predicciones
y_pred_linear_with_outliers = linear_with_outliers.predict(x_test)

# Métricas
mae_linear_with_outliers = mean_absolute_error(y_test, y_pred_linear_with_outliers)
r2_linear_with_outliers = r2_score(y_test, y_pred_linear_with_outliers)

print(f"Linear Regression CON outliers:")
print(f"  - MAE: {mae_linear_with_outliers:.2f}")
print(f"  - R²: {r2_linear_with_outliers:.4f}")

# ============================================================================
# COMPARACIÓN DE MODELOS (FALTA ESTO)
# ============================================================================
print("\n" + "="*60)
print("COMPARACIÓN DE MODELOS")
print("="*60)

# Crear tabla de comparación
comparison_data = {
    'Modelo': ['Linear con outliers', 'Huber con outliers', 'Linear sin outliers'],
    'MAE': [mae_linear_with_outliers, mae_huber_with_outliers, mae_linear_without_outliers],
    'R²': [r2_linear_with_outliers, r2_huber_with_outliers, r2_linear_without_outliers]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Encontrar el mejor modelo por MAE
best_mae_model = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Modelo']
best_mae_value = comparison_df['MAE'].min()

# Encontrar el mejor modelo por R²
best_r2_model = comparison_df.loc[comparison_df['R²'].idxmax(), 'Modelo']
best_r2_value = comparison_df['R²'].max()

print(f"\n📊 RESULTADOS:")
print(f"  - Mejor modelo por MAE: {best_mae_model} (MAE = {best_mae_value:.2f})")
print(f"  - Mejor modelo por R²: {best_r2_model} (R² = {best_r2_value:.4f})")

# ============================================================================
# EVALUACIÓN DE MAE SCORES (FALTA ESTO)
# ============================================================================
print(f"\n🔍 EVALUACIÓN DE MAE SCORES:")
print(f"  - Linear con outliers MAE: {mae_linear_with_outliers:.2f}")
print(f"  - Huber con outliers MAE: {mae_huber_with_outliers:.2f}")
print(f"  - Linear sin outliers MAE: {mae_linear_without_outliers:.2f}")

# Verificar si los últimos dos modelos son mejores que el primero
if mae_huber_with_outliers < mae_linear_with_outliers:
    print(f"  ✅ Huber es mejor que Linear con outliers")
else:
    print(f"  ❌ Huber NO es mejor que Linear con outliers")

if mae_linear_without_outliers < mae_linear_with_outliers:
    print(f"  ✅ Linear sin outliers es mejor que Linear con outliers")
else:
    print(f"  ❌ Linear sin outliers NO es mejor que Linear con outliers")

# ============================================================================
# CONCLUSIÓN (FALTA ESTO)
# ============================================================================
print("\n" + "="*60)
print("CONCLUSIÓN")
print("="*60)

print("Basándonos en los resultados:")

if mae_huber_with_outliers < mae_linear_with_outliers and mae_linear_without_outliers < mae_linear_with_outliers:
    print("✅ Tanto Huber Regression como Linear Regression sin outliers")
    print("   superan al Linear Regression con outliers.")
    print("   Esto confirma que los métodos robustos funcionan mejor")
    print("   cuando hay outliers en los datos.")
elif mae_huber_with_outliers < mae_linear_with_outliers:
    print("✅ Huber Regression supera al Linear Regression con outliers.")
    print("   Esto muestra la robustez del método Huber.")
elif mae_linear_without_outliers < mae_linear_with_outliers:
    print("✅ Linear Regression sin outliers supera al Linear Regression con outliers.")
    print("   Esto confirma que la limpieza de outliers es efectiva.")
else:
    print("❌ Ninguno de los métodos robustos supera al Linear Regression con outliers.")
    print("   Esto podría indicar que los outliers no son tan problemáticos")
    print("   en este dataset específico.")

print(f"\n🎯 Recomendación: Usar {best_mae_model} para este dataset.")
