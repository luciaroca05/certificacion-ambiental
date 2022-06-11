"""
Instalacion de paquetes:
pip install geopandas
pip install Shapely
pip install matplotlib
"""
import pandas as pd # pd es alias de pandas
#from google.colab import drive
import plotly.express as px
import numpy as np
import sklearn as sk

# Carga de datos
df_aprobado = pd.read_json('datasets/Reporte_Proyecto_APROBADO.json')
df_desaprobado = pd.read_json('datasets/Reporte_Proyecto_DESAPROBADO.json')
df_evaluacion = pd.read_json('datasets/Reporte_Proyecto_EN EVALUACION.json')

print(df_aprobado.info())
print()
print(df_desaprobado.info())
print()
print(df_evaluacion.info())

"""
Armar un diccionario de datos de todas las columnas
ID: 
Titular: 
RUC: Registro Único de Contribuyentes
Título Proyecto:
Unidad Proyecto: 
Tipo: 
Actividad:
Fecha de inicio:
Estado:
Descripción:
Longitud y Latitud: Ubicación
Resolución:
Label:

"""

#print(df_aprobado.head())
#print(df_desaprobado.head())
#print(df_evaluacion.head())


# Limpieza, manipulacion y preparacion de datos

# Transformacion del formato de fechas
df_aprobado['FECHA_INICIO'] = df_aprobado['FECHA_INICIO'].astype('datetime64[ns]')
df_desaprobado['FECHA_INICIO'] = df_desaprobado['FECHA_INICIO'].astype('datetime64[ns]')
df_evaluacion['FECHA_INICIO'] = df_evaluacion['FECHA_INICIO'].astype('datetime64[ns]')

df_aprobado['AÑO'] = df_aprobado['FECHA_INICIO'].apply(lambda time: time.year)
df_aprobado['MES'] = df_aprobado['FECHA_INICIO'].apply(lambda time: time.month)
df_aprobado['DIA'] = df_aprobado['FECHA_INICIO'].apply(lambda time: time.day)

df_desaprobado['AÑO'] = df_desaprobado['FECHA_INICIO'].apply(lambda time: time.year)
df_desaprobado['MES'] = df_desaprobado['FECHA_INICIO'].apply(lambda time: time.month)
df_desaprobado['DIA'] = df_desaprobado['FECHA_INICIO'].apply(lambda time: time.day)

df_evaluacion['AÑO'] = df_evaluacion['FECHA_INICIO'].apply(lambda time: time.year)
df_evaluacion['MES'] = df_evaluacion['FECHA_INICIO'].apply(lambda time: time.month)
df_evaluacion['DIA'] = df_evaluacion['FECHA_INICIO'].apply(lambda time: time.day)

# Limpieza de datos
# Identificacion grafica de valores atipicos para latitud, longitud
fig = px.scatter_geo(df_aprobado,lat='LATITUD',lon='LONGITUD', hover_name="ESTADO")
fig.update_layout(title = 'Data de aprobados', title_x=0.5)
fig.show()

# Swap de latitud y longitud mal ingresadas (No hay proyecto en el antártida, por debajo de latitud -74)
df_aprobado['LATITUD'],df_aprobado['LONGITUD']=np.where(df_aprobado['LATITUD']<-74,(df_aprobado['LONGITUD'],df_aprobado['LATITUD']),(df_aprobado['LATITUD'],df_aprobado['LONGITUD']))

# Visualización de datos corregidos
fig = px.scatter_geo(df_aprobado,lat='LATITUD',lon='LONGITUD', hover_name="ESTADO")
fig.update_layout(title = 'Data de aprobados', title_x=0.5)
fig.show()

# Visualización de datos del dataset desaprobados (latitud, longitud)
fig = px.scatter_geo(df_desaprobado,lat='LATITUD',lon='LONGITUD', hover_name="ESTADO")
fig.update_layout(title = 'Data de desaprobados', title_x=0.5)
fig.show()

# Unificación de datas para visualización (aprobados y desaprobados)
df_unificado =  pd.concat([df_aprobado, df_desaprobado])

# Visualización de datos unificados
fig = px.scatter_geo(df_unificado,lat='LATITUD',lon='LONGITUD', hover_name="ESTADO", color="ESTADO")
fig.update_geos(
    showcountries=True
)
fig.update_layout(title = 'Data de aprobados y desaprobados', title_x=0.5)
fig.show()

# Visualización de datos unificados
fig = px.scatter_geo(df_unificado,lat='LATITUD',lon='LONGITUD', hover_name="ESTADO", color="ESTADO")
fig.update_geos(
    showcountries=True
)
fig.update_layout(title = 'Data de aprobados y desaprobados', title_x=0.5)
fig.show()

