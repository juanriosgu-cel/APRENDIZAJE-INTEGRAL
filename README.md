# Aplicación integrada de analítica de talento humano

## Ejecución local
1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar: `streamlit run app.py`

El archivo CSV debe permanecer en la misma carpeta que `app.py`.

## Ajustes aplicados
- Vista previa del resumen traducida al español (etiquetas visuales; nombres internos conservados).
- Clasificación: resultados académicos validados en la tabla principal; Random Forest identificado como modelo final.
- Matriz de confusión de Random Forest fijada a los conteos validados (TN=221, FP=26, FN=26, TP=21).
- Métricas recalculadas, curvas y PCA identificados como análisis exploratorios de la partición actual.
- Campos de predicción individual traducidos y salida de clase/probabilidad explícita.

Regresión y segmentación se conservaron sin cambios intencionales.
