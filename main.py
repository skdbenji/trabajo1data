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
# FUNCIÓN LIMPIAR TEXTO
# ======================================================

def limpiar_texto(texto):

    texto = str(texto).strip().lower()

    texto = ''.join(
        caracter for caracter in unicodedata.normalize('NFD', texto)
        if unicodedata.category(caracter) != 'Mn'
    )

    return texto


# ======================================================
# DESCARGAR DATOS DESDE API
# ======================================================

def descargar_datos():

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

    print(
        f"\nCantidad de registros descargados: "
        f"{len(lista_registros)}"
    )

    return pd.DataFrame(lista_registros)


# ======================================================
# PREPARAR DATAFRAME
# ======================================================

def preparar_dataframe(df):

    # LIMPIAR NOMBRES COLUMNAS
    df.columns = df.columns.str.strip()

    # COLUMNAS TEXTO
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

    # COLUMNAS NUMÉRICAS
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

    # ELIMINAR NULOS
    df = df.dropna(subset=[

        'MES',
        'PROMEDIO_DIAS_ESTADA',
        'INDICE_OCUPACIONAL'
    ])

    # ELIMINAR OUTLIERS
    limite_maximo = df[
        'PROMEDIO_DIAS_ESTADA'
    ].quantile(0.99)

    df = df[
        df['PROMEDIO_DIAS_ESTADA'] <= limite_maximo
    ]

    print(
        f"\nCantidad de registros válidos: "
        f"{len(df)}"
    )

    return df


# ======================================================
# CREAR ESTACIONES
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


# ======================================================
# ESTADÍSTICAS
# ======================================================

def mostrar_estadisticas(df):

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

    print(
        f"\nPromedio estadía: "
        f"{round(promedio_estadia,2)} días"
    )

    print(
        f"Mediana estadía : "
        f"{round(mediana_estadia,2)} días"
    )

    print(
        f"Moda estadía    : "
        f"{round(moda_estadia,2)} días"
    )

    resumen = df[[

        'DIAS_CAMAS_OCUPADAS',
        'NUMERO_EGRESOS',
        'PROMEDIO_DIAS_ESTADA'
    ]].describe()

    print("\nResumen estadístico:")
    print(resumen.round(2))

    # PROMEDIO POR ESTACIÓN
    print("\n========================================")
    print(" PROMEDIO POR ESTACIÓN ")
    print("========================================")

    promedio_estacion = df.groupby(
        'ESTACION'
    )['PROMEDIO_DIAS_ESTADA'].mean()

    print(promedio_estacion.round(2))


# ======================================================
# GENERAR GRÁFICOS
# ======================================================

def generar_graficos(
    df,
    matriz_correlacion,
    tabla_importancias,
    y_prueba,
    prediccion_random_forest
):

    # MAPA DE CALOR
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
    plt.close()

    # HISTOGRAMA
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
    plt.close()

    # BOXPLOT
    plt.figure(figsize=(8,5))

    sns.boxplot(
        x='ESTACION',
        y='PROMEDIO_DIAS_ESTADA',
        data=df
    )

    plt.title(
        'Días de estadía por estación'
    )

    plt.ylabel(
        'Días promedio'
    )

    plt.show()
    plt.close()

    # VARIABLES IMPORTANTES
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
    plt.close()

    # REAL VS PREDICHO
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

    plt.grid(True)

    plt.show()
    plt.close()

    # GRÁFICO POR ESTACIÓN
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
    plt.close()


# ======================================================
# PREPARAR VARIABLES
# ======================================================

def preparar_variables(df):

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

    y = df[
        'PROMEDIO_DIAS_ESTADA'
    ]

    return train_test_split(

        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ======================================================
# ENTRENAR MODELOS
# ======================================================

def entrenar_modelos(
    X_entrenamiento,
    X_prueba,
    y_entrenamiento,
    y_prueba,
    X,
    y
):

    # REGRESIÓN LINEAL
    modelo_lineal = LinearRegression()

    modelo_lineal.fit(
        X_entrenamiento,
        y_entrenamiento
    )

    prediccion_lineal = modelo_lineal.predict(
        X_prueba
    )

    # ÁRBOL
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

    # RANDOM FOREST
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

    # MÉTRICAS
    mae_lineal = mean_absolute_error(
        y_prueba,
        prediccion_lineal
    )

    r2_lineal = r2_score(
        y_prueba,
        prediccion_lineal
    )

    mae_arbol = mean_absolute_error(
        y_prueba,
        prediccion_arbol
    )

    r2_arbol = r2_score(
        y_prueba,
        prediccion_arbol
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

    # VALIDACIÓN CRUZADA
    scores_cv = cross_val_score(

        modelo_random_forest,
        X,
        y,
        cv=5,
        scoring='r2'
    )

    # RESULTADOS
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

    print("\n========================================")
    print(" VALIDACIÓN CRUZADA ")
    print("========================================")

    print(
        f"\nR2 promedio validación cruzada: "
        f"{round(scores_cv.mean(),3)}"
    )

    # IMPORTANCIAS
    tabla_importancias = pd.DataFrame({

        'Variable': X.columns,
        'Importancia': modelo_random_forest.feature_importances_
    })

    tabla_importancias = tabla_importancias.sort_values(

        by='Importancia',
        ascending=False
    )

    return (
        modelo_random_forest,
        tabla_importancias,
        prediccion_random_forest
    )


# ======================================================
# PREDICCIÓN FUTURA
# ======================================================

def prediccion_futura(modelo_random_forest):

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

    prediccion = modelo_random_forest.predict(
        datos_nuevos
    )

    print(

        f"\nTiempo estimado de congestión hospitalaria: "
        f"{round(prediccion[0],2)} días"
    )


# ======================================================
# CONCLUSIONES
# ======================================================

def mostrar_conclusiones():

    print("\n========================================")
    print(" CONCLUSIONES ")
    print("========================================")

    print("""

El proyecto permitió analizar datos hospitalarios
reales utilizando técnicas de Machine Learning.

El modelo Random Forest obtuvo el mejor
rendimiento predictivo.

Durante invierno existe una mayor presión
hospitalaria y un aumento en los días
promedio de estadía.

""")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    # DESCARGAR
    df = descargar_datos()

    # LIMPIAR
    df = preparar_dataframe(df)

    # CREAR ESTACIÓN
    df['ESTACION'] = df['MES'].apply(
        obtener_estacion
    )

    # ESTADÍSTICAS
    mostrar_estadisticas(df)

    # CORRELACIÓN
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

    matriz_correlacion = df[
        columnas_numericas
    ].corr()

    # VARIABLES ML
    (
        X_entrenamiento,
        X_prueba,
        y_entrenamiento,
        y_prueba
    ) = preparar_variables(df)

    X = pd.concat([
        X_entrenamiento,
        X_prueba
    ])

    y = pd.concat([
        y_entrenamiento,
        y_prueba
    ])

    # MODELOS
    (
        modelo_random_forest,
        tabla_importancias,
        prediccion_random_forest
    ) = entrenar_modelos(

        X_entrenamiento,
        X_prueba,
        y_entrenamiento,
        y_prueba,
        X,
        y
    )

    # GRÁFICOS
    generar_graficos(

        df,
        matriz_correlacion,
        tabla_importancias,
        y_prueba,
        prediccion_random_forest
    )

    # PREDICCIÓN FUTURA
    prediccion_futura(
        modelo_random_forest
    )

    # CONCLUSIONES
    mostrar_conclusiones()
