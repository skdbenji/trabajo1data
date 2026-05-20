import pandas as pd

# ======================================================
# URLS ORIGINALES (NO BORRAR)
# ======================================================

# url_2023 = "https://github.com/skdbenji/trabajo1data/releases/download/v1.0.0/AtencionesUrgencia2023.csv"
# url_2024 = "https://github.com/skdbenji/trabajo1data/releases/download/v1.0.0/AtencionesUrgencia2024.csv"

# ======================================================
# ARCHIVOS MINI DE PRUEBA
# ======================================================

url_2023 = "AtencionesUrgenciaMini.csv"
url_2024 = "AtencionesUrgenciaMini.csv"

print("Conectando y cargando datos...")

# LEER CSV COMPLETO PARA VER NOMBRES DE COLUMNAS
df_test = pd.read_csv(
    url_2023,
    encoding="latin1",
    sep=';'
)

# MOSTRAR COLUMNAS DISPONIBLES
print("\nColumnas encontradas en el CSV:")
print(df_test.columns)

# ======================================================
# COLUMNAS A UTILIZAR
# ======================================================

col_fecha = "fecha"
col_Atenciones = "Total"

columnas_a_cargar = [col_fecha, col_Atenciones]

# ======================================================
# CARGA DE DATOS
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

print("\nDatos cargados con éxito")

# ======================================================
# UNIR DATASETS
# ======================================================

df_completo = pd.concat([df_2023, df_2024], ignore_index=True)

# ======================================================
# LIMPIEZA Y CONVERSIÓN DE DATOS
# ======================================================

df_completo[col_fecha] = pd.to_datetime(
    df_completo[col_fecha],
    errors='coerce'
)

df_completo[col_Atenciones] = pd.to_numeric(
    df_completo[col_Atenciones],
    errors='coerce'
)

# ELIMINAR FILAS VACÍAS
df_completo = df_completo.dropna(
    subset=[col_fecha, col_Atenciones]
)

# ORDENAR POR FECHA
df_completo = df_completo.sort_values(
    by=col_fecha,
    ascending=True
)

print("\nClasificando registros por estación...")

# ======================================================
# FUNCIÓN PARA ASIGNAR ESTACIONES
# ======================================================

def Asignar_Estacion(fecha):

    mes = fecha.month
    dia = fecha.day

    # VERANO
    if (mes == 12 and dia >= 21) or (mes in [1, 2]) or (mes == 3 and dia < 21):
        return 'Verano'

    # OTOÑO
    elif (mes == 3 and dia >= 21) or (mes in [4, 5]) or (mes == 6 and dia < 21):
        return 'Otoño'

    # INVIERNO
    elif (mes == 6 and dia >= 21) or (mes in [7, 8]) or (mes == 9 and dia < 21):
        return 'Invierno'

    # PRIMAVERA
    else:
        return 'Primavera'

# ======================================================
# CREAR COLUMNA ESTACIÓN
# ======================================================

df_completo['Estacion'] = df_completo[col_fecha].apply(Asignar_Estacion)

# ======================================================
# ESTADÍSTICAS DESCRIPTIVAS
# ======================================================

print("\n" + "=" * 50)
print(" Estadística Descriptiva por Estación ")
print("=" * 50)

estadisticas_por_estacion = df_completo.groupby('Estacion')[col_Atenciones].agg(
    Casos_Totales='count',
    Promedio='mean',
    Mediana='median',
    Desviacion_Estandar='std',
    Minimo='min',
    Maximo='max'
)

# ======================================================
# MOSTRAR RESULTADOS
# ======================================================

print("\n")
print(estadisticas_por_estacion.round(2))
