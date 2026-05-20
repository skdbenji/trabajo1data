#Este es el repostiorio del nuevo trabajo


url_2023="https://github.com/skdbenji/trabajo1data/releases/download/v1.0.0/AtencionesUrgencia2024.csv"
url_2024="https://github.com/skdbenji/trabajo1data/releases/download/v1.0.0/AtencionesUrgencia2024.csv"

col_fecha = "fecha"
col_Atenciones = "total"

print("Conectando y cargando datos")

columnas_a_cargar = [col_fecha, col_Atenciones]

df_2023 = pd.read_csv(url_2023, usecols=columnas_a_cargar)
df_2024 = pd.read_csv(url_2024, usecols=columnas_a_cargar)

print("Datos cargados con exito")

df_completo = pd.concat([df_2023, df_2024], ignore_index=True)

df_completo[col_fecha] = pd.to_datetime(df_completo[col_fecha])
df_completo[col_Atenciones] = pd.to_numeric(pf_completo[col_Atenciones], arrors='coerce')

df_completo = df_completo.dropna(subset=[col_Atenciones])

df_completo = df_completo.sort_values(by=col_fecha, ascending=True)

print("Clasificando registros de urgencia por estacion")

def Asignar_Estacion(fecha):
  mes = fecha.month
  dia = fecha.day

  if(mes == 12 and dia >= 21) or (mes in [1, 2]) or (mes == 3 and dia < 21):
    return 'Verano'

  elif (mes == 3 and dia >= 21) or (mes in [4, 5]) or (mes == 6 and dia < 21):
    return 'Otoño'

  elif (mes == 6 and dia >= 21) or (mes in [7,8]) or (mes == 9 and dia < 21):
    return 'Invierno'

  else:
    return 'Primavera'


df_completo['Estacion'] = df_completo[col_fecha].apply(Asignar_Estacion)


print("\n" + "="*50)
print("  Estadistica Descriptiva por Estacion")
print("="*50)

estadisticas_por_estacion = df_completo.groupby('Estacion')[col_Atenciones].agg(
  casos_Totales='count',
  Promedios='mean',
  Mediana='var',
  desciacion_Estandar='std',
  Minimo='min',
  Maximo='max'
)

print(estadisticas_por_estacion.round(2))
  
