# Analítica de talento humano — versión corregida

Incluye la aplicación integrada con clasificación, regresión del ingreso, segmentación, EDA y metodología.

## Archivos
- `app.py`: aplicación Streamlit corregida.
- `WA_Fn-UseC_-HR-Employee-Attrition.csv`: dataset requerido.
- `requirements.txt`: dependencias.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Ajustes realizados
- Traducción de las 35 variables de IBM HR Analytics en etiquetas visibles de clasificación; los nombres internos se conservan para compatibilidad.
- Random Forest queda como modelo final seleccionado del estudio, sin ocultar las métricas comparativas recalculadas en la app.
- Regresión: interpretación de R², MAE, RMSE, coeficientes y limitaciones; Ridge se presenta como alternativa ligeramente más robusta según el documento académico.
- Segmentación: visualización PCA de los grupos, conteos y porcentajes por clúster, perfiles de medias e interpretación descriptiva por grupo para K-means, jerárquico y DBSCAN.

Nota: las métricas calculadas por la aplicación pueden diferir de las reportadas en los documentos/notebooks por particiones, parámetros o procedimientos distintos.
