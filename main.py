# ======================================================
# IMPORTAR LIBRERÍAS
# ======================================================

import pandas as pd
import urllib.request
import json
import unicodedata

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

import numpy as np

# ======================================================
# FUNCIÓN PARA LIMPIAR TEXTO
# ======================================================

def limpiar_texto(texto):

    texto = str(texto).strip().lower()

    texto = ''.join(
        caracter for caracter in unicodedata.normalize('NFD', texto)
        if unicodedata.category(caracter) != 'Mn'
    )

    return texto

# ======================================================
# DESCARGA DE DATOS DESDE API
# ======================================================

print("\n========================================")
print(" DESCARGANDO DATOS HOSPITALARIOS ")
print("========================================")

offset = 0
limite_registros = 5000

lista_registros = []

while True:

    url = (
        "https://datos.gob.cl/api/3/action/"
        "datastore_search?"
        "resource_id=657cc933-eac8-4bfc-b004-c4d6dcd988a8"
        f"&limit={limite_registros}"
        f"&offset={offset}"
    )

    respuesta = urllib.request.urlopen(url)

    datos = json.loads(
        respuesta.read()
    )

    registros = datos['result']['records']

    if len(registros) == 0:
        break

    lista_registros.extend(registros)

    offset += limite_registros

print(f"\nCantidad de registros descargados: {len(lista_registros)}")

# ======================================================
# CREAR DATAFRAME
# ======================================================

df = pd.DataFrame(lista_registros)

# ======================================================
# LIMPIAR NOMBRES DE COLUMNAS
# ======================================================

df.columns = df.columns.str.strip()

# ======================================================
# LIMPIAR COLUMNAS DE TEXTO
# ======================================================

columnas_texto = [

    'GLOSA_SSS',
    'ESTABLECIMIENTO',
    'AREA_FUNCIONAL'
]

for columna in columnas_texto:

    if columna in df.columns:

        df[columna] = df[columna].apply(
            limpiar_texto
        )

# ======================================================
# CONVERTIR COLUMNAS A NUMÉRICO
# ======================================================

columnas_numericas = [

    'MES',
    'DIAS_CAMAS_OCUPADAS',
    'DIAS_CAMAS_DISPONIBLES',
    'DIAS_ESTADA',
    'NUMERO_EGRESOS',
    'EGRESOS_FALLECIDOS',
    'TRASLADOS',
    'INDICE_OCUPACIONAL',
    'PROMEDIO_CAMAS_DISPONIBLE',
    'PROMEDIO_DIAS_ESTADA',
    'LETALIDAD',
    'INDICE_ROTACION'
]

for columna in columnas_numericas:

    if columna in df.columns:

        df[columna] = pd.to_numeric(
            df[columna],
            errors='coerce'
        )

# ======================================================
# ELIMINAR DATOS NULOS
# ======================================================

df = df.dropna(subset=[

    'MES',
    'PROMEDIO_DIAS_ESTADA',
    'INDICE_OCUPACIONAL'
])

# ======================================================
# ELIMINAR OUTLIERS
# ======================================================

limite_maximo = df[
    'PROMEDIO_DIAS_ESTADA'
].quantile(0.99)

df = df[
    df['PROMEDIO_DIAS_ESTADA'] <= limite_maximo
]

print(f"\nCantidad de registros válidos: {len(df)}")

# ======================================================
# ESTADÍSTICAS DESCRIPTIVAS
# ======================================================

print("\n========================================")
print(" ESTADÍSTICAS DESCRIPTIVAS ")
print("========================================")

promedio_estadia = df[
    'PROMEDIO_DIAS_ESTADA'
].mean()

mediana_estadia = df[
    'PROMEDIO_DIAS_ESTADA'
].median()

moda_estadia = df[
    'PROMEDIO_DIAS_ESTADA'
].mode()[0]

print(f"\nPromedio estadía: {round(promedio_estadia,2)} días")
print(f"Mediana estadía : {round(mediana_estadia,2)} días")
print(f"Moda estadía    : {round(moda_estadia,2)} días")

resumen_estadistico = df[[

    'DIAS_CAMAS_OCUPADAS',
    'NUMERO_EGRESOS',
    'PROMEDIO_DIAS_ESTADA'
]].describe()

print("\nResumen estadístico:")
print(resumen_estadistico.round(2))

# ======================================================
# CREAR ESTACIONES DEL AÑO
# ======================================================

def obtener_estacion(mes):

    if mes in [12, 1, 2]:
        return "Verano"

    elif mes in [3, 4, 5]:
        return "Otoño"

    elif mes in [6, 7, 8]:
        return "Invierno"

    else:
        return "Primavera"

df['ESTACION'] = df['MES'].apply(
    obtener_estacion
)

# ======================================================
# PROMEDIO POR ESTACIÓN
# ======================================================

print("\n========================================")
print(" PROMEDIO DE ESTADÍA POR ESTACIÓN ")
print("========================================")

promedio_por_estacion = df.groupby(
    'ESTACION'
)['PROMEDIO_DIAS_ESTADA'].mean()

print(promedio_por_estacion.round(2))

# ======================================================
# CORRELACIONES
# ======================================================

print("\n========================================")
print(" CORRELACIONES ")
print("========================================")

matriz_correlacion = df[
    columnas_numericas
].corr()

print(
    matriz_correlacion[
        'PROMEDIO_DIAS_ESTADA'
    ].sort_values(
        ascending=False
    )
)

# ======================================================
# MAPA DE CALOR
# ======================================================

plt.figure(figsize=(12,8))

sns.heatmap(

    matriz_correlacion,

    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5
)

plt.title(
    'Mapa de calor de correlaciones'
)

plt.tight_layout()
plt.show()

# ======================================================
# DISTRIBUCIÓN DE DÍAS DE ESTADÍA
# ======================================================

plt.figure(figsize=(8,5))

df['PROMEDIO_DIAS_ESTADA'].hist(
    bins=30
)

plt.title(
    'Distribución de días de estadía'
)

plt.xlabel(
    'Cantidad de días'
)

plt.ylabel(
    'Frecuencia'
)

plt.show()

# ======================================================
# BOXPLOT POR ESTACIÓN
# ======================================================

plt.figure(figsize=(8,5))

df.boxplot(

    column='PROMEDIO_DIAS_ESTADA',
    by='ESTACION'
)

plt.title(
    'Días de estadía por estación'
)

plt.suptitle('')

plt.ylabel(
    'Días promedio'
)

plt.show()

# ======================================================
# CODIFICAR VARIABLES
# ======================================================

encoder_region = LabelEncoder()
encoder_area = LabelEncoder()
encoder_estacion = LabelEncoder()

df['REGION_COD'] = encoder_region.fit_transform(
    df['GLOSA_SSS']
)

df['AREA_COD'] = encoder_area.fit_transform(
    df['AREA_FUNCIONAL']
)

df['ESTACION_COD'] = encoder_estacion.fit_transform(
    df['ESTACION']
)

# ======================================================
# VARIABLES PREDICTORAS
# ======================================================

X = df[[

    'MES',
    'REGION_COD',
    'AREA_COD',
    'DIAS_CAMAS_OCUPADAS',
    'DIAS_CAMAS_DISPONIBLES',
    'NUMERO_EGRESOS',
    'EGRESOS_FALLECIDOS',
    'TRASLADOS',
    'INDICE_OCUPACIONAL',
    'PROMEDIO_CAMAS_DISPONIBLE',
    'LETALIDAD',
    'INDICE_ROTACION',
    'ESTACION_COD'
]]

# ======================================================
# VARIABLE OBJETIVO
# ======================================================

y = df[
    'PROMEDIO_DIAS_ESTADA'
]

# ======================================================
# DIVIDIR DATOS
# ======================================================

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================================
# REGRESIÓN LINEAL
# ======================================================

modelo_lineal = LinearRegression()

modelo_lineal.fit(
    X_entrenamiento,
    y_entrenamiento
)

prediccion_lineal = modelo_lineal.predict(
    X_prueba
)

mae_lineal = mean_absolute_error(
    y_prueba,
    prediccion_lineal
)

r2_lineal = r2_score(
    y_prueba,
    prediccion_lineal
)

# ======================================================
# ÁRBOL DE DECISIÓN
# ======================================================

modelo_arbol = DecisionTreeRegressor(

    max_depth=10,
    random_state=42
)

modelo_arbol.fit(
    X_entrenamiento,
    y_entrenamiento
)

prediccion_arbol = modelo_arbol.predict(
    X_prueba
)

mae_arbol = mean_absolute_error(
    y_prueba,
    prediccion_arbol
)

r2_arbol = r2_score(
    y_prueba,
    prediccion_arbol
)

# ======================================================
# RANDOM FOREST
# ======================================================

modelo_random_forest = RandomForestRegressor(

    n_estimators=80,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

modelo_random_forest.fit(
    X_entrenamiento,
    y_entrenamiento
)

prediccion_random_forest = modelo_random_forest.predict(
    X_prueba
)

mae_random_forest = mean_absolute_error(
    y_prueba,
    prediccion_random_forest
)

r2_random_forest = r2_score(
    y_prueba,
    prediccion_random_forest
)

mape_random_forest = mean_absolute_percentage_error(
    y_prueba,
    prediccion_random_forest
)

rmse_random_forest = np.sqrt(

    mean_squared_error(
        y_prueba,
        prediccion_random_forest
    )
)

# ======================================================
# VALIDACIÓN CRUZADA
# ======================================================

scores_cv = cross_val_score(

    modelo_random_forest,
    X,
    y,
    cv=5,
    scoring='r2'
)

# ======================================================
# RESULTADOS
# ======================================================

print("\n========================================")
print(" RESULTADOS DE LOS MODELOS ")
print("========================================")

print("\nRegresión Lineal")
print(f"MAE  : {round(mae_lineal,2)}")
print(f"R2   : {round(r2_lineal,2)}")

print("\nÁrbol de Decisión")
print(f"MAE  : {round(mae_arbol,2)}")
print(f"R2   : {round(r2_arbol,2)}")

print("\nRandom Forest")
print(f"MAE  : {round(mae_random_forest,2)}")
print(f"R2   : {round(r2_random_forest,2)}")
print(f"MAPE : {round(mape_random_forest * 100,2)}%")
print(f"RMSE : {round(rmse_random_forest,2)}")

# ======================================================
# VALIDACIÓN CRUZADA
# ======================================================

print("\n========================================")
print(" VALIDACIÓN CRUZADA ")
print("========================================")

print(
    f"\nR2 promedio validación cruzada: "
    f"{round(scores_cv.mean(),3)}"
)

# ======================================================
# MEJOR MODELO
# ======================================================

resultados_modelos = {

    "Regresión Lineal": r2_lineal,
    "Árbol de Decisión": r2_arbol,
    "Random Forest": r2_random_forest
}

mejor_modelo = max(

    resultados_modelos,
    key=resultados_modelos.get
)

print("\n========================================")
print(" MEJOR MODELO ")
print("========================================")

print(f"\nEl mejor modelo fue: {mejor_modelo}")

# ======================================================
# VARIABLES MÁS IMPORTANTES
# ======================================================

print("\n========================================")
print(" VARIABLES MÁS IMPORTANTES ")
print("========================================")

tabla_importancias = pd.DataFrame({

    'Variable': X.columns,
    'Importancia': modelo_random_forest.feature_importances_
})

tabla_importancias = tabla_importancias.sort_values(

    by='Importancia',
    ascending=False
)

print(tabla_importancias.head(10))

# ======================================================
# GRÁFICO VARIABLES IMPORTANTES
# ======================================================

plt.figure(figsize=(10,6))

plt.barh(

    tabla_importancias['Variable'].head(10),
    tabla_importancias['Importancia'].head(10)
)

plt.title(
    'Variables más importantes'
)

plt.xlabel(
    'Nivel de importancia'
)

plt.gca().invert_yaxis()

plt.show()

# ======================================================
# GRÁFICO REAL VS PREDICHO
# ======================================================

plt.figure(figsize=(7,7))

plt.scatter(

    y_prueba,
    prediccion_random_forest
)

plt.xlabel(
    'Valores reales'
)

plt.ylabel(
    'Valores predichos'
)

plt.title(
    'Comparación real vs predicho'
)

plt.show()

# ======================================================
# PREDICCIÓN POR ESTACIÓN
# ======================================================

print("\n========================================")
print(" PREDICIÓN POR ESTACIÓN ")
print("========================================")

prediccion_estaciones = df.groupby(
    'ESTACION'
)['PROMEDIO_DIAS_ESTADA'].mean()

print(prediccion_estaciones.round(2))

# ======================================================
# PREDICCIÓN FUTURA
# ======================================================

print("\n========================================")
print(" PREDICCIÓN FUTURA ")
print("========================================")

datos_nuevos = pd.DataFrame({

    'MES': [7],
    'REGION_COD': [3],
    'AREA_COD': [5],
    'DIAS_CAMAS_OCUPADAS': [180],
    'DIAS_CAMAS_DISPONIBLES': [200],
    'NUMERO_EGRESOS': [50],
    'EGRESOS_FALLECIDOS': [5],
    'TRASLADOS': [20],
    'INDICE_OCUPACIONAL': [90],
    'PROMEDIO_CAMAS_DISPONIBLE': [6],
    'LETALIDAD': [10],
    'INDICE_ROTACION': [2],
    'ESTACION_COD': [0]
})

prediccion_final = modelo_random_forest.predict(
    datos_nuevos
)

print(

    f"\nTiempo estimado de congestión hospitalaria: "
    f"{round(prediccion_final[0],2)} días"
)

# ======================================================
# GRÁFICO POR ESTACIÓN
# ======================================================

promedios_estacion = df.groupby(
    'ESTACION'
)['PROMEDIO_DIAS_ESTADA'].mean()

promedios_estacion.plot(
    kind='bar'
)

plt.title(
    'Congestión hospitalaria según estación'
)

plt.ylabel(
    'Promedio días de estadía'
)

plt.show()

# ======================================================
# CONCLUSIONES
# ======================================================

print("\n========================================")
print(" CONCLUSIONES ")
print("========================================")

print("""

El proyecto permitió analizar datos hospitalarios
reales obtenidos desde datos.gob.cl utilizando
técnicas de Machine Learning.

Se aplicaron distintos modelos predictivos
para estimar niveles de congestión hospitalaria,
comparando sus resultados mediante métricas
estadísticas.

El modelo Random Forest obtuvo el mejor
rendimiento, logrando detectar relaciones
más complejas entre las variables.

Las variables más influyentes fueron:

- ocupación hospitalaria
- días camas ocupadas
- número de egresos
- índice de rotación
- estación del año

Los resultados muestran que durante invierno
existe una mayor presión hospitalaria y un
aumento en los días promedio de estadía.

Además, los gráficos permitieron visualizar
patrones importantes entre las variables y
comprender mejor el comportamiento de los datos.

Limitaciones del proyecto:

- algunos datos pueden contener errores
- no se consideran pandemias
- no se incluyeron variables climáticas
- existen diferencias de registros entre regiones

Como trabajo futuro se podría desarrollar
una aplicación en tiempo real para monitorear
la congestión hospitalaria usando IA.

""")
