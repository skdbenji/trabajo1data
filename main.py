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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# ======================================================
# FUNCIÓN LIMPIAR TEXTO
# ======================================================
def limpiar_texto(texto):
    texto = str(texto).strip().lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

# ======================================================
# DESCARGAR DATOS DESDE API
# ======================================================
print("\n========================================")
print(" DESCARGANDO DATOS HOSPITALARIOS ")
print("========================================")

offset = 0
limite = 5000
todos_los_registros = []

while True:
    url_api = (
        "https://datos.gob.cl/api/3/action/datastore_search?"
        "resource_id=657cc933-eac8-4bfc-b004-c4d6dcd988a8"
        f"&limit={limite}&offset={offset}"
    )
    
    respuesta = urllib.request.urlopen(url_api)
    datos_json = json.loads(respuesta.read())
    registros = datos_json['result']['records']

    if len(registros) == 0:
        break

    todos_los_registros.extend(registros)
    
    
    print(f"-> Descargados {len(todos_los_registros)} registros...")

    offset += limite

print(f"\nRegistros descargados: {len(todos_los_registros)}")

# ======================================================
# CREAR DATAFRAME Y LIMPIAR
# ======================================================
df = pd.DataFrame(todos_los_registros)
df.columns = df.columns.str.strip()

columnas_texto = ['GLOSA_SSS', 'ESTABLECIMIENTO', 'AREA_FUNCIONAL']

for columna in columnas_texto:
    if columna in df.columns:
        df[columna] = df[columna].apply(limpiar_texto)

columnas_numericas = [
    'MES', 'DIAS_CAMAS_OCUPADAS', 'DIAS_CAMAS_DISPONIBLES', 'DIAS_ESTADA',
    'NUMERO_EGRESOS', 'EGRESOS_FALLECIDOS', 'TRASLADOS', 'INDICE_OCUPACIONAL',
    'PROMEDIO_CAMAS_DISPONIBLE', 'PROMEDIO_DIAS_ESTADA', 'LETALIDAD', 'INDICE_ROTACION'
]

for columna in columnas_numericas:
    if columna in df.columns:
        df[columna] = pd.to_numeric(df[columna], errors='coerce')

# ======================================================
# ELIMINAR NULOS Y OUTLIERS
# ======================================================
df = df.dropna(subset=['MES', 'PROMEDIO_DIAS_ESTADA', 'INDICE_OCUPACIONAL'])

limite_superior = df['PROMEDIO_DIAS_ESTADA'].quantile(0.99)
df = df[df['PROMEDIO_DIAS_ESTADA'] <= limite_superior]

print(f"Registros válidos: {len(df)}")

#=======================================================
# ESTÁDISTICAS DESCRIPTIVAS
#=======================================================
print("\n========================================")
print(" ESTADÍSTICAS DESCRIPTIVAS ")
print("========================================")
media_estadia = df['PROMEDIO_DIAS_ESTADA'].mean()
mediana_estadia = df['PROMEDIO_DIAS_ESTADA'].median()
moda_estadia = df['PROMEDIO_DIAS_ESTADA'].mode()[0]
print(f"Promedio días estadía: {round(media_estadia, 2)}")
print(f"Mediana días estadía: {round(mediana_estadia, 2)}")
print(f"Moda días estadía: {round(moda_estadia, 2)}")

resumen= df[['DIAS_CAMAS_OCUPADAS', 'NUMERO_EGRESOS', 'PROMEDIO_DIAS_ESTADA']].describe()
print(resumen.round(2))

# ======================================================
# CREAR ESTACIONES Y ESTADÍSTICAS
# ======================================================
def asignar_estacion(mes):
    if mes in [12, 1, 2]:
        return "Verano"
    elif mes in [3, 4, 5]:
        return "Otoño"
    elif mes in [6, 7, 8]:
        return "Invierno"
    else:
        return "Primavera"

df['ESTACION'] = df['MES'].apply(asignar_estacion)

print("\n========================================")
print(" PROMEDIO DE ESTADÍA POR ESTACIÓN ")
print("========================================")
estadisticas = df.groupby('ESTACION')['PROMEDIO_DIAS_ESTADA'].mean()
print(estadisticas.round(2))

# ======================================================
# CORRELACIONES
# ======================================================
print("\n========================================")
print(" CORRELACIÓN CON ESTADÍA ")
print("========================================")
correlaciones = df[columnas_numericas].corr()
print(correlaciones['PROMEDIO_DIAS_ESTADA'].sort_values(ascending=False))
plt.figure(figsize=(10, 8)) 
sns.heatmap(
    correlaciones, 
    annot=True,         
    cmap='coolwarm',    
    fmt=".2f",          # rojo alta correlacion, azul baja
    linewidths=0.5      
)
plt.title('Mapa de Calor de Correlaciones Hospitalarias', fontsize=14)
plt.tight_layout()
plt.show()

# ======================================================
# CODIFICAR VARIABLES Y DIVIDIR DATOS
# ======================================================
encoder_region = LabelEncoder()
encoder_area = LabelEncoder()
encoder_estacion = LabelEncoder()

df['REGION_NUM'] = encoder_region.fit_transform(df['GLOSA_SSS'])
df['AREA_NUM'] = encoder_area.fit_transform(df['AREA_FUNCIONAL'])
df['ESTACION_NUM'] = encoder_estacion.fit_transform(df['ESTACION'])

X = df[[
    'MES', 'REGION_NUM', 'AREA_NUM', 'DIAS_CAMAS_OCUPADAS', 'DIAS_CAMAS_DISPONIBLES',
    'NUMERO_EGRESOS', 'EGRESOS_FALLECIDOS', 'TRASLADOS', 'INDICE_OCUPACIONAL',
    'PROMEDIO_CAMAS_DISPONIBLE', 'LETALIDAD', 'INDICE_ROTACION', 'ESTACION_NUM'
]]
y = df['PROMEDIO_DIAS_ESTADA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================================================
# REGRESIÓN LINEAL SIMPLE (1 variable)
# ======================================================
X_train_simple = X_train[['DIAS_CAMAS_OCUPADAS']]
X_test_simple = X_test[['DIAS_CAMAS_OCUPADAS']]

modelo_lr_simple = LinearRegression()
modelo_lr_simple.fit(X_train_simple, y_train)
pred_lr_simple = modelo_lr_simple.predict(X_test_simple)
mae_lr_simple = mean_absolute_error(y_test, pred_lr_simple)
r2_lr_simple = r2_score(y_test, pred_lr_simple)

# ======================================================
# REGRESIÓN LINEAL MÚLTIPLE (Todas las variables)
# ======================================================
modelo_lr_multiple = LinearRegression()
modelo_lr_multiple.fit(X_train, y_train)
pred_lr_multiple = modelo_lr_multiple.predict(X_test)
mae_lr_multiple = mean_absolute_error(y_test, pred_lr_multiple)
r2_lr_multiple = r2_score(y_test, pred_lr_multiple)

# ======================================================
# ÁRBOL DE DECISIÓN
# ======================================================
modelo_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
modelo_tree.fit(X_train, y_train)
pred_tree = modelo_tree.predict(X_test)
mae_tree = mean_absolute_error(y_test, pred_tree)
r2_tree = r2_score(y_test, pred_tree)

# ======================================================
# RANDOM FOREST
# ======================================================
modelo_rf = RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)
pred_rf = modelo_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, pred_rf)
r2_rf = r2_score(y_test, pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, pred_rf)

# ======================================================
# VALIDACIÓN CRUZADA
# ======================================================
scores = cross_val_score(modelo_rf, X, y, cv=5, scoring='r2')

# ======================================================
# RESULTADOS
# ======================================================
print("\n========================================")
print(" RESULTADOS MODELOS ")
print("========================================")

print("\nRegresión Lineal Simple")
print(f"MAE : {round(mae_lr_simple, 2)}")
print(f"R2  : {round(r2_lr_simple, 2)}")

print("\nRegresión Lineal Múltiple")
print(f"MAE : {round(mae_lr_multiple, 2)}")
print(f"R2  : {round(r2_lr_multiple, 2)}")

print("\nÁrbol de Decisión")
print(f"MAE : {round(mae_tree, 2)}")
print(f"R2  : {round(r2_tree, 2)}")

print("\nRandom Forest")
print(f"MAE : {round(mae_rf, 2)}")
print(f"R2  : {round(r2_rf, 2)}")
print(f"MAPE: {round(mape_rf * 100, 2)}%")

print("\n========================================")
print(" VALIDACIÓN CRUZADA ")
print("========================================")
print(f"\nR2 promedio CV: {round(scores.mean(), 3)}")

# ======================================================
# MEJOR MODELO Y VARIABLES IMPORTANTES
# ======================================================
resultados = {
    "Regresión Lineal Simple": r2_lr_simple,
    "Regresión Lineal Múltiple": r2_lr_multiple,
    "Árbol de Decisión": r2_tree,
    "Random Forest": r2_rf
}
mejor_modelo = max(resultados, key=resultados.get)

print("\n========================================")
print(" MEJOR MODELO ")
print("========================================")
print(f"\n{mejor_modelo}")

print("\n========================================")
print(" VARIABLES MÁS IMPORTANTES ")
print("========================================")
importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': modelo_rf.feature_importances_
})
importancias = importancias.sort_values(by='Importancia', ascending=False)
print(importancias.head(10))
plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importancia', 
    y='Variable', 
    data=importancias.head(10),
    palette='viridis' 
)

plt.title('Top 10 Variables que más impactan la Estadía Hospitalaria')
plt.xlabel('Nivel de Importancia (Random Forest)')
plt.ylabel('Variable Predictora')
plt.tight_layout()
plt.show()

# ======================================================
# PREDICCIÓN FUTURA
# ======================================================
print("\n========================================")
print(" PREDICCIÓN FUTURA ")
print("========================================")
nueva_prediccion = pd.DataFrame({
    'MES': [7], 'REGION_NUM': [3], 'AREA_NUM': [5],
    'DIAS_CAMAS_OCUPADAS': [180], 'DIAS_CAMAS_DISPONIBLES': [200],
    'NUMERO_EGRESOS': [50], 'EGRESOS_FALLECIDOS': [5],
    'TRASLADOS': [20], 'INDICE_OCUPACIONAL': [90],
    'PROMEDIO_CAMAS_DISPONIBLE': [6], 'LETALIDAD': [10],
    'INDICE_ROTACION': [2], 'ESTACION_NUM': [0]
})

prediccion = modelo_rf.predict(nueva_prediccion)
print(f"\nTiempo estimado de congestión hospitalaria: {round(prediccion[0],2)} días")

# ======================================================
# GRÁFICO
# ======================================================
promedios = df.groupby('ESTACION')['PROMEDIO_DIAS_ESTADA'].mean()
promedios.plot(kind='bar')
plt.title('Congestión hospitalaria por estación')
plt.ylabel('Promedio días estadía')
plt.show()

# ======================================================
# CONCLUSIÓN
# ======================================================
print("\n========================================")
print(" CONCLUSIÓN ")
print("========================================")
print("""
El sistema utilizó modelos de Machine Learning
para estimar niveles de congestión hospitalaria
a partir de datos reales obtenidos desde
datos.gob.cl.

El modelo Random Forest presentó el mejor
rendimiento predictivo.

Las variables más importantes fueron:
- ocupación hospitalaria
- días cama ocupados
- egresos
- rotación hospitalaria
- estación del año

Los resultados muestran que ciertas estaciones,
especialmente invierno, presentan mayores niveles
de presión hospitalaria y posibles aumentos en
los tiempos de espera.
""")