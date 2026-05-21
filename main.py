# ======================================================
# IMPORTAR LIBRERÍAS
# ======================================================

import pandas as pd
import unicodedata

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

# ======================================================
# FUNCIÓN PARA LIMPIAR TEXTO
# ======================================================

def limpiar_texto(texto):

    texto = str(texto).strip().lower()

    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

    return texto

# ======================================================
# URLS ORIGINALES
# ======================================================

# url_2023 = "https://github.com/skdbenji/trabajo1data/releases/download/v1.0.0/AtencionesUrgencia2023.csv"

# url_2024 = "https://github.com/skdbenji/trabajo1data/releases/download/v1.0.0/AtencionesUrgencia2024.csv"

# ======================================================
# ARCHIVOS MINI DE PRUEBA
# ======================================================

url_2023 = "AtencionesUrgenciaMini.csv"
url_2024 = "AtencionesUrgenciaMini.csv"

print("Conectando y cargando datos...")

# ======================================================
# LEER CSV PARA VER COLUMNAS
# ======================================================

df_test = pd.read_csv(
    url_2023,
    encoding="latin1",
    sep=';'
)

print("\nColumnas encontradas:")
print(df_test.columns)

# ======================================================
# COLUMNAS A UTILIZAR
# ======================================================

col_fecha = "fecha"

col_atenciones = "Total"

columnas_a_cargar = [
    col_fecha,
    col_atenciones,
    'NombreRegion',
    'NombreComuna'
]

# ======================================================
# CARGAR DATOS
# ======================================================

df_2023 = pd.read_csv(
    url_2023,
    usecols=columnas_a_cargar,
    encoding="latin1",
    sep=';'
)

df_2024 = pd.read_csv(
    url_2024,
    usecols=columnas_a_cargar,
    encoding="latin1",
    sep=';'
)

print("\nDatos cargados correctamente")

# ======================================================
# UNIR DATASETS
# ======================================================

df_completo = pd.concat(
    [df_2023, df_2024],
    ignore_index=True
)

# ======================================================
# LIMPIEZA DE DATOS
# ======================================================

print("\nIniciando limpieza de datos...")

# ======================================================
# ELIMINAR ESPACIOS EN COLUMNAS
# ======================================================

df_completo.columns = df_completo.columns.str.strip()

# ======================================================
# LIMPIAR TEXTO REGIONES Y COMUNAS
# ======================================================

df_completo['NombreRegion'] = df_completo[
    'NombreRegion'
].apply(limpiar_texto)
# ======================================================
# REGIONES VÁLIDAS DE CHILE
# ======================================================
regiones_validas = [

    'arica y parinacota',

    'tarapaca',

    'antofagasta',

    'atacama',

    'coquimbo',

    'valparaiso',

    'metropolitana',

    'ohiggins',

    'maule',

    'nuble',

    'biobio',

    'araucania',

    'rios',

    'lagos',

    'aysen',

    'magallanes y de la antartica chilena'
]
# NORMALIZAR NOMBRES DE REGIONES

df_completo['NombreRegion'] = df_completo[
    'NombreRegion'
].str.replace('de la ', '', regex=False)

df_completo['NombreRegion'] = df_completo[
    'NombreRegion'
].str.replace('de los ', '', regex=False)

df_completo['NombreRegion'] = df_completo[
    'NombreRegion'
].str.replace('de ', '', regex=False)

df_completo['NombreRegion'] = df_completo[
    'NombreRegion'
].str.replace('del ', '', regex=False)

df_completo['NombreRegion'] = df_completo[
    'NombreRegion'
].str.replace('region ', '', regex=False)

df_completo['NombreRegion'] = df_completo[
    'NombreRegion'
].str.strip()

# ======================================================
# FILTRAR SOLO REGIONES VÁLIDAS
# ======================================================

df_completo = df_completo[
    df_completo['NombreRegion']
    .isin(regiones_validas)
]

df_completo['NombreComuna'] = df_completo[
    'NombreComuna'
].apply(limpiar_texto)

# ======================================================
# CONVERTIR FECHAS
# ======================================================

df_completo[col_fecha] = pd.to_datetime(
    df_completo[col_fecha],
    errors='coerce',
    dayfirst=True
)

# ======================================================
# CONVERTIR A NUMÉRICO
# ======================================================

df_completo[col_atenciones] = pd.to_numeric(
    df_completo[col_atenciones],
    errors='coerce'
)

# ======================================================
# MOSTRAR NULOS
# ======================================================

print("\nValores nulos encontrados:")
print(df_completo.isnull().sum())

# ======================================================
# ELIMINAR FILAS VACÍAS
# ======================================================

df_completo = df_completo.dropna(
    subset=[
        col_fecha,
        col_atenciones,
        'NombreRegion',
        'NombreComuna'
    ]
)

# ======================================================
# ELIMINAR VALORES NEGATIVOS
# ======================================================

df_completo = df_completo[
    df_completo[col_atenciones] >= 0
]

# ======================================================
# ELIMINAR OUTLIERS EXTREMOS
# ======================================================

limite_superior = df_completo[
    col_atenciones
].quantile(0.99)

df_completo = df_completo[
    df_completo[col_atenciones] <= limite_superior
]

# ======================================================
# VALIDAR RANGO DE AÑOS
# ======================================================

df_completo = df_completo[
    (df_completo[col_fecha].dt.year >= 2023)
    &
    (df_completo[col_fecha].dt.year <= 2024)
]

# ======================================================
# ELIMINAR COMUNAS VACÍAS
# ======================================================

df_completo = df_completo[
    df_completo['NombreComuna'].str.len() > 2
]

# ======================================================
# ORDENAR POR FECHA
# ======================================================

df_completo = df_completo.sort_values(
    by=col_fecha
)

print("\nLimpieza completada")

print("\nCantidad de registros:")
print(len(df_completo))

# ======================================================
# FUNCIÓN PARA ASIGNAR ESTACIONES
# ======================================================

print("\nClasificando registros por estación...")

def asignar_estacion(fecha):

    mes = fecha.month
    dia = fecha.day

    # VERANO
    if (
        (mes == 12 and dia >= 21)
        or (mes in [1, 2])
        or (mes == 3 and dia < 21)
    ):
        return 'Verano'

    # OTOÑO
    elif (
        (mes == 3 and dia >= 21)
        or (mes in [4, 5])
        or (mes == 6 and dia < 21)
    ):
        return 'Otoño'

    # INVIERNO
    elif (
        (mes == 6 and dia >= 21)
        or (mes in [7, 8])
        or (mes == 9 and dia < 21)
    ):
        return 'Invierno'

    # PRIMAVERA
    else:
        return 'Primavera'

# ======================================================
# CREAR COLUMNA ESTACIÓN
# ======================================================

df_completo['Estacion'] = df_completo[
    col_fecha
].apply(asignar_estacion)

# ======================================================
# ESTADÍSTICA DESCRIPTIVA
# ======================================================

print("\n" + "=" * 60)
print(" ESTADÍSTICA DESCRIPTIVA ")
print("=" * 60)

print("\n Metricas globales del Dataset:")
resumen_gobal = df_completo[col_atenciones].describe()
asimetria = df_completo[col_atenciones].skew()
curtosis = df_completo[col_atenciones].kurt()

print(resumen_global.round(2))
print(f"Asimetria: {asimetria:2f} (valores > 0 implican sesgo a la derecha)")
print(f"Curtosis: {curtosis:2f} (valores altos indican presencia de eventos atipicos o brotes)")

print("\n Analisis descriptivo por estacion del año: ")
estadisticas_estacion = df_completo.groupby(
    'Estacion'
)[col_atenciones].agg(

    Casos_Totales='count',

    Promedio='mean',

    Mediana='median',

    Varianza='var',

    Desviacion_Estandar='std',

    Minimo='min',

    Maximo='max'
)

print("\n")
print(estadisticas_estacion.round(2))

#ESTADISTICAS COMPARATIVA POR AÑO
print("\n Comparativa temporal 2023 vs 2024: ")
df_completo['Año_Temporal'] = df_completo[col_fecha].dt.year
estadisticas_ano = df_completo.groupby('Año_Temporal')[col_atenciones].agg(
    Total_Atenciones='sum',
    Promedio_Diario='mean',
    Maximo_en_un_Dia='max'
)
print(estadisticas_ano.round(2))

print("\n Top 5 Regiones con mayor promedio de atencion de urgencia: ")
estadisticas_region = df_completo.groupby('NombreRegion')[col_atenciones].mean().sort_values(ascending=False).head(5)
print(estadisticas_region.round(2))

# ======================================================
# MACHINE LEARNING
# ======================================================

print("\n" + "=" * 60)
print(" MODELOS PREDICTIVOS ")
print("=" * 60)

# ======================================================
# VARIABLES TEMPORALES
# ======================================================

df_completo['Mes'] = df_completo[
    col_fecha
].dt.month

df_completo['Dia'] = df_completo[
    col_fecha
].dt.day

df_completo['Año'] = df_completo[
    col_fecha
].dt.year

# ======================================================
# CODIFICAR ESTACIÓN
# ======================================================

encoder = LabelEncoder()

df_completo['Estacion_Num'] = encoder.fit_transform(
    df_completo['Estacion']
)

# ======================================================
# VARIABLES PREDICTORAS Y OBJETIVO
# ======================================================

X = df_completo[
    ['Mes', 'Dia', 'Año', 'Estacion_Num']
]

y = df_completo[col_atenciones]

# ======================================================
# VALIDAR CANTIDAD DE DATOS
# ======================================================

print("\nCantidad de muestras para ML:")
print(len(X))

# ======================================================
# DIVIDIR DATOS
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================================
# REGRESIÓN LINEAL
# ======================================================

print("\n" + "=" * 40)
print(" REGRESIÓN LINEAL ")
print("=" * 40)

modelo_lr = LinearRegression()

modelo_lr.fit(X_train, y_train)

pred_lr = modelo_lr.predict(X_test)

mae_lr = mean_absolute_error(
    y_test,
    pred_lr
)

r2_lr = r2_score(
    y_test,
    pred_lr
)

print("MAE :", round(mae_lr, 2))
print("R2  :", round(r2_lr, 2))

# ======================================================
# ÁRBOL DE DECISIÓN
# ======================================================

print("\n" + "=" * 40)
print(" ÁRBOL DE DECISIÓN ")
print("=" * 40)

modelo_tree = DecisionTreeRegressor(
    random_state=42
)

modelo_tree.fit(X_train, y_train)

pred_tree = modelo_tree.predict(X_test)

mae_tree = mean_absolute_error(
    y_test,
    pred_tree
)

r2_tree = r2_score(
    y_test,
    pred_tree
)

print("MAE :", round(mae_tree, 2))
print("R2  :", round(r2_tree, 2))

# ======================================================
# RANDOM FOREST
# ======================================================

print("\n" + "=" * 40)
print(" RANDOM FOREST ")
print("=" * 40)

modelo_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

modelo_rf.fit(X_train, y_train)

pred_rf = modelo_rf.predict(X_test)

mae_rf = mean_absolute_error(
    y_test,
    pred_rf
)

r2_rf = r2_score(
    y_test,
    pred_rf
)

print("MAE :", round(mae_rf, 2))
print("R2  :", round(r2_rf, 2))

# ======================================================
# COMPARACIÓN FINAL
# ======================================================

print("\n" + "=" * 60)
print(" COMPARACIÓN FINAL ")
print("=" * 60)

print("\nRegresión Lineal")
print("MAE :", round(mae_lr, 2))
print("R2  :", round(r2_lr, 2))

print("\nÁrbol de Decisión")
print("MAE :", round(mae_tree, 2))
print("R2  :", round(r2_tree, 2))

print("\nRandom Forest")
print("MAE :", round(mae_rf, 2))
print("R2  :", round(r2_rf, 2))

# ======================================================
# MEJOR MODELO
# ======================================================

resultados = {
    "Regresión Lineal": r2_lr,
    "Árbol de Decisión": r2_tree,
    "Random Forest": r2_rf
}

mejor_modelo = max(
    resultados,
    key=resultados.get
)

print("\n" + "=" * 60)
print(" MEJOR MODELO ")
print("=" * 60)

print("\nEl modelo con mejor rendimiento fue:")

print(mejor_modelo)
