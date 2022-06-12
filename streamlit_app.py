"""
git add . && git commit -m "" && git push
"""
# Importación de bibliotecas
import pandas as pd
import plotly.express as px
import numpy as np
import sklearn as sk
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt

# Carga de datos
df_aprobado = pd.read_json('datasets/Reporte_Proyecto_APROBADO.json')
df_desaprobado = pd.read_json('datasets/Reporte_Proyecto_DESAPROBADO.json')
df_evaluacion = pd.read_json('datasets/Reporte_Proyecto_EN EVALUACION.json')

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

# Swap de latitud y longitud mal ingresadas (No hay proyecto en el antártida, por debajo de latitud -74)
df_aprobado['LATITUD'],df_aprobado['LONGITUD']=np.where(df_aprobado['LATITUD']<-74,(df_aprobado['LONGITUD'],df_aprobado['LATITUD']),(df_aprobado['LATITUD'],df_aprobado['LONGITUD']))

# Reemplazo de valoes duplicados en la columna ACTIVIDAD (Residuos Sólidos y Transporte)
df_aprobado['ACTIVIDAD'] = df_aprobado['ACTIVIDAD'].replace('Residuos Solidos','Residuos Sólidos')
df_aprobado['ACTIVIDAD'] = df_aprobado['ACTIVIDAD'].replace('Transportes','Transporte')

df_desaprobado['ACTIVIDAD'] = df_desaprobado['ACTIVIDAD'].replace('Residuos Solidos','Residuos Sólidos')
df_desaprobado['ACTIVIDAD'] = df_desaprobado['ACTIVIDAD'].replace('Transportes','Transporte')

df_evaluacion['ACTIVIDAD'] = df_evaluacion['ACTIVIDAD'].replace('Residuos Solidos','Residuos Sólidos')
df_evaluacion['ACTIVIDAD'] = df_evaluacion['ACTIVIDAD'].replace('Transportes','Transporte')

# Unificación de datas para visualización (aprobados y desaprobados)
df_unificado =  pd.concat([df_aprobado, df_desaprobado])


#App
st.title('Certificación Ambiental - Proyecto Final')
st.markdown("---") # Linea divisoria
st.markdown("##") # Linea en blanco
st.write('Bienvenidx al **app**')


#Proyectos con fecha
#st.write('Proyectos por fecha')

#fecha_inicio = st.slider("Ver informacion registrada en:",
#    value = datetime(2020,1,1,9,30),
#    format = "DD/MM/YY - hh:mm")
#st.write("Fecha seleccionada:", fecha_inicio)

# Seleccion del dataset
st.write('Seleccionar proyecto por estado')
opcion_dataset = st.selectbox(
     '¿Qué dataset deseas visualizar?',
     ('Proyectos aprobados', 'Proyectos desaprobados', 'Proyectos en evaluacion'))
df_visualizacion = None
estado = '-'
if opcion_dataset == 'Proyectos aprobados':
    df_visualizacion = df_aprobado
    estado = 'aprobados'
elif opcion_dataset == 'Proyectos desaprobados':
    df_visualizacion = df_desaprobado
    estado = 'desaprobados'
elif opcion_dataset == 'Proyectos en evaluacion':
    df_visualizacion = df_evaluacion
    estado = 'en evaluación'

st.write('Seleccionó visualizar',len(df_visualizacion.index),'proyectos',estado,'en total:')
# Agregar  un widget quemuestre metadatos (cols, rows, no nulls, etc)



# Cruces de datos plot(x,y)
df_actividad_freq = pd.DataFrame(df_visualizacion["ACTIVIDAD"].value_counts())
labels = df_actividad_freq.index.tolist()
sizes = df_actividad_freq["ACTIVIDAD"].tolist()
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
startangle=0)
plt.title('Distribucion de datos segun ACTIVIDAD')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig1)

df_tipo_freq = pd.DataFrame(df_visualizacion["TIPO"].value_counts())
labels = df_tipo_freq.index.tolist()
sizes = df_tipo_freq["TIPO"].tolist()
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
startangle=0, textprops={'fontsize': 10})
plt.title('Distribucion de datos segun TIPO')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig1)

# Agrega titulo con write
df_anho_freq = pd.DataFrame(df_visualizacion["AÑO"].value_counts())
st.bar_chart(df_anho_freq)

#print(df_actividad_freq.head())
#st.dataframe(df_actividad_freq)
#st.bar_chart(df_anho_freq)

# Visualizacion de datos en el mapa
st.write('Ubicación de los proyectos',estado)
df_visualizacion = df_visualizacion.rename(columns={'LATITUD':'lat', 'LONGITUD':'lon'})
st.map(df_visualizacion[['lat','lon']])

# Tabla de datos (ver si se puede agregar widget de busqueda)
st.write('Tabla de datos',estado,'en formato DataFrame')
st.dataframe(df_visualizacion)



