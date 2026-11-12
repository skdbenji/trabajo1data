# 🏥 Predicción de días de estada hospitalaria en Chile

Modelo de machine learning que predice el **promedio de días de estada** de los pacientes en hospitales públicos de Chile a partir de los indicadores de ocupación de camas. La estada promedio se usa como señal de **congestión hospitalaria**: cuando sube, las camas se liberan más lento y el sistema se satura.

Los datos son abiertos y se descargan de [datos.gob.cl](https://datos.gob.cl) cada vez que se ejecuta el proyecto.

Proyecto académico de Ciencia de Datos, Ingeniería en Informática, UTEM.

## Pregunta

¿Se puede estimar cuántos días se quedará en promedio un paciente en un área de un hospital, sabiendo cómo está la ocupación de camas, los egresos y la época del año?

## Flujo del proyecto

```
datos.gob.cl (API CKAN) → limpieza → variables → estadística descriptiva → modelos → evaluación → predicción
```

1. **Descarga automática.** Consulta la API `datastore_search` de datos.gob.cl con paginación de 5.000 registros hasta traer el conjunto completo.
2. **Limpieza.** Normaliza el texto (tildes, mayúsculas y espacios), convierte las columnas numéricas, elimina filas con nulos en las variables clave y recorta los valores extremos sobre el percentil 99.
3. **Ingeniería de variables.**
   - Estación del año a partir del mes.
   - Codificación del servicio de salud, del área funcional y de la estación (`LabelEncoder`).
4. **Estadística descriptiva.** Resumen de las variables y estada promedio por estación del año.
5. **Modelado** (dos enfoques, ver abajo).
6. **Gráficos** de resultados y **predicción** para un escenario hipotético.

## Variables

**Objetivo:** `PROMEDIO_DIAS_ESTADA`

**Predictoras (13):** mes, servicio de salud, área funcional, estación, días-cama ocupados, días-cama disponibles, número de egresos, egresos fallecidos, traslados, índice ocupacional, promedio de camas disponibles, letalidad e índice de rotación.

## Modelos

### `main.py`: modelos clásicos (scikit-learn)

| Modelo | Configuración |
|---|---|
| Regresión lineal | Línea base |
| Árbol de decisión | `max_depth=10` |
| Random Forest | 80 árboles, `max_depth=12` |

- División entrenamiento/prueba 80/20 (`random_state=42`).
- Métricas: **MAE, RMSE, MAPE y R²** sobre el conjunto de prueba.
- **Validación cruzada de 5 folds** (R²) para revisar que el resultado no dependa de una sola partición.

### `main_redes_neuronales.py`: red neuronal (TensorFlow/Keras)

- Red densa 64 → 32 → 16 → 1 con activación ReLU y **Dropout del 30 %** en las dos primeras capas.
- Variables estandarizadas con `StandardScaler`.
- Optimizador Adam (tasa de aprendizaje 0,001), pérdida MSE, hasta 100 épocas con lotes de 32.
- **EarlyStopping** (paciencia 15) para evitar sobreajuste y **ReduceLROnPlateau** (factor 0,5, paciencia 7) para ajustar la tasa de aprendizaje.
- Mismas métricas que los modelos clásicos, para compararlos directamente.

## Tecnologías

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · matplotlib · seaborn

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py                    # modelos clásicos
python main_redes_neuronales.py   # red neuronal
```

Se necesita conexión a internet: los datos se descargan al ejecutar.

## Estructura

```
main.py                     descarga, limpieza, modelos clásicos, gráficos y predicción
main_redes_neuronales.py    mismo flujo de datos con una red neuronal en Keras
AtencionesUrgenciaMini.csv  muestra de atenciones de urgencia usada como referencia
requirements.txt            dependencias
```

## Limitaciones

- La predicción de escenarios usa valores ingresados a mano; no es un pronóstico oficial.
- Los códigos de servicio de salud y área funcional se codifican como números, lo que es suficiente para modelos de árboles pero no ideal para la regresión lineal.
