# ======================================================
# IMPORTAR LIBRERÍAS
# ======================================================

import pandas as pd
import urllib.request
import json
import unicodedata

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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
        f"{round(promedio_estadia, 2)} días"
    )

    print(
        f"Mediana estadía : "
        f"{round(mediana_estadia, 2)} días"
    )

    print(
        f"Moda estadía    : "
        f"{round(moda_estadia, 2)} días"
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
# PREPARAR VARIABLES
# ======================================================

def preparar_variables(df):

    encoder_region  = LabelEncoder()
    encoder_area    = LabelEncoder()
    encoder_estacion = LabelEncoder()

    df['REGION_COD']   = encoder_region.fit_transform(df['GLOSA_SSS'])
    df['AREA_COD']     = encoder_area.fit_transform(df['AREA_FUNCIONAL'])
    df['ESTACION_COD'] = encoder_estacion.fit_transform(df['ESTACION'])

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

    y = df['PROMEDIO_DIAS_ESTADA']

    # DIVIDIR EN ENTRENAMIENTO Y PRUEBA
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # NORMALIZAR CARACTERÍSTICAS
    # Las redes neuronales requieren datos normalizados para
    # un entrenamiento estable y evitar que variables con
    # escalas grandes dominen el gradiente.
    scaler = StandardScaler()

    X_entrenamiento_norm = scaler.fit_transform(X_entrenamiento)
    X_prueba_norm        = scaler.transform(X_prueba)

    return (
        X_entrenamiento_norm,
        X_prueba_norm,
        y_entrenamiento,
        y_prueba,
        scaler
    )


# ======================================================
# CONSTRUIR RED NEURONAL
# ======================================================

def construir_red(n_entradas):

    # ARQUITECTURA:
    # Capa de entrada → Capa oculta 1 (64 neuronas, ReLU, Dropout)
    #                → Capa oculta 2 (32 neuronas, ReLU, Dropout)
    #                → Capa oculta 3 (16 neuronas, ReLU)
    #                → Capa de salida (1 neurona, lineal)
    #
    # ReLU: evita el problema del gradiente desvaneciente.
    # Dropout(0.3): desactiva aleatoriamente el 30% de las neuronas
    #              durante el entrenamiento para reducir el sobreajuste.
    # BatchNormalization: estabiliza y acelera el entrenamiento.
    # Salida lineal: apropiada para regresión (predecir días continuos).

    modelo = keras.Sequential([

        # CAPA DE ENTRADA + PRIMERA CAPA OCULTA
        layers.Dense(
            64,
            activation='relu',
            input_shape=(n_entradas,),
            name='capa_oculta_1'
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # SEGUNDA CAPA OCULTA
        layers.Dense(
            32,
            activation='relu',
            name='capa_oculta_2'
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # TERCERA CAPA OCULTA
        layers.Dense(
            16,
            activation='relu',
            name='capa_oculta_3'
        ),

        # CAPA DE SALIDA (regresión → activación lineal)
        layers.Dense(
            1,
            activation='linear',
            name='capa_salida'
        )
    ])

    # COMPILAR EL MODELO
    # Optimizador Adam: combina momentum y tasa de aprendizaje adaptativa.
    # Función de pérdida MSE: penaliza más los errores grandes.
    # Métrica MAE: más interpretable (en días reales).
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return modelo


# ======================================================
# ENTRENAR RED NEURONAL
# ======================================================

def entrenar_red(
    modelo,
    X_entrenamiento,
    y_entrenamiento,
    X_prueba,
    y_prueba
):

    print("\n========================================")
    print(" ENTRENANDO RED NEURONAL ")
    print("========================================")

    print("\nArquitectura del modelo:")
    modelo.summary()

    # CALLBACKS
    # EarlyStopping: detiene el entrenamiento si la pérdida de validación
    #               no mejora en 15 épocas seguidas, evitando sobreajuste.
    # ReduceLROnPlateau: reduce la tasa de aprendizaje si la pérdida
    #                   se estanca, para afinar el entrenamiento.
    parada_temprana = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reducir_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )

    # ENTRENAMIENTO
    historial = modelo.fit(
        X_entrenamiento,
        y_entrenamiento,
        epochs=100,
        batch_size=32,
        validation_data=(X_prueba, y_prueba),
        callbacks=[parada_temprana, reducir_lr],
        verbose=1
    )

    return historial


# ======================================================
# EVALUAR RED NEURONAL
# ======================================================

def evaluar_red(modelo, X_prueba, y_prueba):

    print("\n========================================")
    print(" RESULTADOS DE LA RED NEURONAL ")
    print("========================================")

    predicciones = modelo.predict(X_prueba).flatten()

    mae  = mean_absolute_error(y_prueba, predicciones)
    rmse = np.sqrt(mean_squared_error(y_prueba, predicciones))
    mape = mean_absolute_percentage_error(y_prueba, predicciones)
    r2   = r2_score(y_prueba, predicciones)

    print(f"\nMAE  : {round(mae, 2)} días")
    print(f"RMSE : {round(rmse, 2)} días")
    print(f"MAPE : {round(mape * 100, 2)}%")
    print(f"R²   : {round(r2, 4)}")

    return predicciones


# ======================================================
# GENERAR GRÁFICOS
# ======================================================

def generar_graficos(df, historial, y_prueba, predicciones):

    # HISTOGRAMA DE DÍAS DE ESTADÍA
    plt.figure(figsize=(8, 5))
    df['PROMEDIO_DIAS_ESTADA'].hist(bins=30)
    plt.title('Distribución de días de estadía')
    plt.xlabel('Cantidad de días')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()
    plt.close()

    # BOXPLOT POR ESTACIÓN
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='ESTACION', y='PROMEDIO_DIAS_ESTADA', data=df)
    plt.title('Días de estadía por estación')
    plt.ylabel('Días promedio')
    plt.tight_layout()
    plt.show()
    plt.close()

    # CURVA DE PÉRDIDA (LOSS) — ENTRENAMIENTO VS VALIDACIÓN
    # Esta curva permite visualizar si el modelo está aprendiendo
    # correctamente o si sufre sobreajuste (overfitting).
    plt.figure(figsize=(10, 5))
    plt.plot(historial.history['loss'],     label='Pérdida entrenamiento')
    plt.plot(historial.history['val_loss'], label='Pérdida validación')
    plt.title('Curva de pérdida (MSE) durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    # CURVA DE MAE — ENTRENAMIENTO VS VALIDACIÓN
    plt.figure(figsize=(10, 5))
    plt.plot(historial.history['mae'],     label='MAE entrenamiento')
    plt.plot(historial.history['val_mae'], label='MAE validación')
    plt.title('Curva de MAE durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Error absoluto medio (MAE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    # REAL VS PREDICHO
    plt.figure(figsize=(7, 7))
    plt.scatter(y_prueba, predicciones, alpha=0.5)
    minimo = min(y_prueba.min(), predicciones.min())
    maximo = max(y_prueba.max(), predicciones.max())
    plt.plot([minimo, maximo], [minimo, maximo], 'r--', label='Predicción perfecta')
    plt.xlabel('Valores reales')
    plt.ylabel('Valores predichos')
    plt.title('Comparación real vs predicho — Red Neuronal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    # GRÁFICO POR ESTACIÓN
    promedios_estacion = df.groupby('ESTACION')['PROMEDIO_DIAS_ESTADA'].mean()
    promedios_estacion.plot(kind='bar')
    plt.title('Congestión hospitalaria según estación')
    plt.ylabel('Promedio días de estadía')
    plt.tight_layout()
    plt.show()
    plt.close()

    # HISTOGRAMA DE ERRORES DE PREDICCIÓN
    errores = y_prueba.values - predicciones
    plt.figure(figsize=(8, 5))
    plt.hist(errores, bins=30, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Error cero')
    plt.title('Distribución de errores de predicción')
    plt.xlabel('Error (real − predicho)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


# ======================================================
# PREDICCIÓN FUTURA
# ======================================================

def prediccion_futura(modelo, scaler):

    print("\n========================================")
    print(" PREDICCIÓN FUTURA ")
    print("========================================")

    # Se usan los mismos valores de ejemplo que en el trabajo
    # de Random Forest para comparar los resultados entre modelos.
    datos_nuevos = pd.DataFrame({
        'MES':                     [7],
        'REGION_COD':              [3],
        'AREA_COD':                [5],
        'DIAS_CAMAS_OCUPADAS':     [180],
        'DIAS_CAMAS_DISPONIBLES':  [200],
        'NUMERO_EGRESOS':          [50],
        'EGRESOS_FALLECIDOS':      [5],
        'TRASLADOS':               [20],
        'INDICE_OCUPACIONAL':      [90],
        'PROMEDIO_CAMAS_DISPONIBLE': [6],
        'LETALIDAD':               [10],
        'INDICE_ROTACION':         [2],
        'ESTACION_COD':            [0]
    })

    # Normalizar con el mismo scaler del entrenamiento
    datos_norm = scaler.transform(datos_nuevos)

    prediccion = modelo.predict(datos_norm).flatten()

    print(
        f"\nTiempo estimado de congestión hospitalaria: "
        f"{round(prediccion[0], 2)} días"
    )


# ======================================================
# CONCLUSIONES
# ======================================================

def mostrar_conclusiones():

    print("\n========================================")
    print(" CONCLUSIONES ")
    print("========================================")

    print("""

El proyecto adaptó el análisis hospitalario de Random Forest
a una Red Neuronal Artificial (MLP) con Keras/TensorFlow.

Arquitectura utilizada:
  · Capa oculta 1: 64 neuronas, ReLU, BatchNorm, Dropout(0.3)
  · Capa oculta 2: 32 neuronas, ReLU, BatchNorm, Dropout(0.3)
  · Capa oculta 3: 16 neuronas, ReLU
  · Capa de salida: 1 neurona, activación lineal (regresión)

Decisiones de diseño:
  · StandardScaler: normalización obligatoria para redes neuronales.
  · EarlyStopping: evita sobreajuste deteniendo el entrenamiento
    cuando la pérdida de validación deja de mejorar.
  · ReduceLROnPlateau: ajusta la tasa de aprendizaje dinámicamente.
  · BatchNormalization: estabiliza y acelera la convergencia.

Durante invierno se mantiene una mayor presión hospitalaria
y un aumento en los días promedio de estadía.

""")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    # REPRODUCIBILIDAD
    tf.random.set_seed(42)
    np.random.seed(42)

    # DESCARGAR
    df = descargar_datos()

    # LIMPIAR
    df = preparar_dataframe(df)

    # CREAR ESTACIÓN
    df['ESTACION'] = df['MES'].apply(obtener_estacion)

    # ESTADÍSTICAS
    mostrar_estadisticas(df)

    # PREPARAR VARIABLES (con normalización para la red)
    (
        X_entrenamiento,
        X_prueba,
        y_entrenamiento,
        y_prueba
    ) = preparar_variables(df)[:4]

    # Recuperar scaler para usarlo en predicción futura
    (
        X_entrenamiento,
        X_prueba,
        y_entrenamiento,
        y_prueba,
        scaler
    ) = preparar_variables(df)

    # CONSTRUIR RED NEURONAL
    n_entradas = X_entrenamiento.shape[1]
    modelo = construir_red(n_entradas)

    # ENTRENAR
    historial = entrenar_red(
        modelo,
        X_entrenamiento,
        y_entrenamiento,
        X_prueba,
        y_prueba
    )

    # EVALUAR
    predicciones = evaluar_red(modelo, X_prueba, y_prueba)

    # GRÁFICOS
    generar_graficos(df, historial, y_prueba, predicciones)

    # PREDICCIÓN FUTURA
    prediccion_futura(modelo, scaler)

    # CONCLUSIONES
    mostrar_conclusiones()
