# 🏥 Predicción de días de estada hospitalaria en Chile

Modelo que predice el **promedio de días de estada** de los pacientes en hospitales públicos de Chile a partir de indicadores de ocupación de camas, usando datos abiertos de [datos.gob.cl](https://datos.gob.cl).

Proyecto académico de Ciencia de Datos — Ingeniería en Informática, UTEM.

## Qué hace

- **Descarga automática** de los datos desde la API de datos.gob.cl, paginada de 5.000 en 5.000 registros.
- **Limpieza**: normaliza texto (tildes y mayúsculas), convierte columnas numéricas, elimina nulos y outliers (percentil 99).
- **Ingeniería de variables**: estación del año a partir del mes, codificación de establecimiento, servicio de salud y área funcional.
- **Estadística descriptiva** y promedios por estación.
- **Dos enfoques de modelado**:
  - `main.py`: regresión lineal, árbol de decisión y Random Forest, comparados con validación cruzada (MAE, RMSE, MAPE, R²).
  - `main_redes_neuronales.py`: red neuronal densa en Keras/TensorFlow con early stopping y ajuste de learning rate.
- **Gráficos** de resultados y **predicción** para escenarios futuros.

## Tecnologías

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · matplotlib · seaborn

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py                    # modelos clásicos
python main_redes_neuronales.py   # red neuronal
```

Se necesita conexión a internet: los datos se descargan al ejecutar.
