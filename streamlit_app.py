"""
https://github.com/luciaroca05/proyecto-final-certificacion-ambiental/
https://share.streamlit.io/luciaroca05/proyecto-final-certificacion-ambiental/main
cd Desktop
cd proy (TAB)
streamlit run streamlit_app.py
git pull && git add . && git commit -m "." && git push

"""

#----------------------------------INSTALACIÓN----E----IMPORTACIÓN--------------------------------------

# Importación de bibliotecas
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

# Librerias y modulos de ML, DA
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Excel
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

#------------------------------------------------MACHINE----LEARNING--------------------------------------------

# Obtención de datos

## Carga de datos
df_aprobado = pd.read_json('datasets/Reporte_Proyecto_APROBADO.json')
df_desaprobado = pd.read_json('datasets/Reporte_Proyecto_DESAPROBADO.json')
df_evaluacion = pd.read_json('datasets/Reporte_Proyecto_EN EVALUACION.json')

# Limpiar, Preparar y Manipular Data

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

## Swap de latitud y longitud mal ingresadas (No hay proyecto en el antártida, por debajo de latitud -74)
df_aprobado['LATITUD'],df_aprobado['LONGITUD']=np.where(df_aprobado['LATITUD']<-74,(df_aprobado['LONGITUD'],df_aprobado['LATITUD']),(df_aprobado['LATITUD'],df_aprobado['LONGITUD']))

## Reemplazo de valores duplicados en la columna ACTIVIDAD (Residuos Sólidos y Transporte)
df_aprobado['ACTIVIDAD'] = df_aprobado['ACTIVIDAD'].replace('Residuos Solidos','Residuos Sólidos')
df_aprobado['ACTIVIDAD'] = df_aprobado['ACTIVIDAD'].replace('Transportes','Transporte')

df_desaprobado['ACTIVIDAD'] = df_desaprobado['ACTIVIDAD'].replace('Residuos Solidos','Residuos Sólidos')
df_desaprobado['ACTIVIDAD'] = df_desaprobado['ACTIVIDAD'].replace('Transportes','Transporte')

df_evaluacion['ACTIVIDAD'] = df_evaluacion['ACTIVIDAD'].replace('Residuos Solidos','Residuos Sólidos')
df_evaluacion['ACTIVIDAD'] = df_evaluacion['ACTIVIDAD'].replace('Transportes','Transporte')

## Unificación de datas para visualización (aprobados y desaprobados)
df_unificado =  pd.concat([df_aprobado, df_desaprobado])


# Preparacion de datos

## Dataframe aprobado se divide en 2: part1 (800) y parte2 (153)
df_aprobado_parte1  = df_aprobado.sample(n = 800, random_state=1) # Esto para entrenamiento del modelo
df_aprobado_parte2 = df_aprobado.drop(df_aprobado_parte1.index)

## Reseteo de los indices de los datsets generados

df_aprobado_parte1.reset_index()
df_aprobado_parte2.reset_index()

## Dataframe desaprobado se divide en 2: part1 (80) y parte2 (12)

df_desaprobado_parte1  = df_desaprobado.sample(n = 80, random_state=1)
df_desaprobado_parte2 = df_desaprobado.drop(df_desaprobado_parte1.index)

## Reseteo de los indices de los datsets generados

df_desaprobado_parte1.reset_index()
df_desaprobado_parte2.reset_index()

## Repetir 10 veces los 80 datos de la parte 1 del dataset desprobado = 10x80 = 800

df_desaprobado_parte1 =  pd.concat([df_desaprobado_parte1]*10) # Esto para entrenamiento del modelo

## Creacion del dataframe para entrenamiento (train) y test del modelo

df_train =  pd.concat([df_aprobado_parte1, df_desaprobado_parte1])
df_test = pd.concat([df_aprobado_parte2, df_desaprobado_parte2])

df_train.reset_index()
df_test.reset_index()

## Separacion de columnas X y columna y para a partir de los dataframe de train y test

y_train = df_train.iloc[:,8]
X_train = df_train.iloc[:,[10,11,14,15]]# elección de columnas para predicción (Longitud, latitud, año, mes)

y_test = df_test.iloc[:,8]
X_test = df_test.iloc[:,[10,11,14,15]]# elección de columnas para predicción (Longitud, latitud, año, mes)


# Entrenamiento y test de los modelos para la predicción

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
SVM = svm.SVC(decision_function_shape="ovo").fit(X_train, y_train)
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1).fit(X_train, y_train)

#-----------------------------------------CÓDIGOS---DEL---APP---------------------------------------------------

#App

#Título
st.markdown("<h1 style='text-align: center; color: black;'>Certificación Ambiental - Proyecto Final</h1>", unsafe_allow_html=True)
st.markdown("##") # Linea en blanco

#Bienvenida
st.markdown("<h1 style='text-align: center; color: black;'>Bienvenidx a nuestra APP 😄!</h1>", unsafe_allow_html=True)
st.markdown("---") # Linea divisoria
st.markdown("##") # Linea en blanco

#Introducción al grupo
st.header('Nosotras')
st.markdown("##") # Linea en blanco
st.caption('Grupo de estudiantes del 5to ciclo de la carrera de Ingeniería Ambiental de UPCH del curso de Programación Avanzada')

#Integrantes con foto
image = Image.open('integrantes.jpg')
st.image(image, caption='Dona Nicole Chancan Aviles, María Fernanda Cisneros Moron, Sharon Nicolle Dextre Cartolin, Daniella Mercedes Palacios Li, Lucía Fernanda Roca Cuadros')
st.markdown("##") # Linea en blanco

st.markdown("---") # Linea divisoria

#Sobre el APP
st.header('Sobre el APP')
st.markdown("##") # Linea en blanco
'''
En esta APP se va a relizar el análisis, visualización y clasificación, utilizando datos sobre certificación
ambiental del Servicio Nacional de Certificación Ambiental para las Inversiones Sostenibles – SENACE.

Fuente: https://www.datosabiertos.gob.pe/dataset/certificaci%C3%B3n-ambiental

Los datos de los proyectos están clasificados en las siguientes categorías,
'''
st.write('ID: ') 
st.write('Titular: ')
st.write('RUC: Registro Único de Contribuyentes')
st.write('Título Proyecto: ')
st.write('Unidad Proyecto: ')
st.write('Tipo: ')
st.write('Actividad: ')
st.write('Fecha de inicio: ')
st.write('Estado: ')
st.write('Descripción: ')
st.write('Longitud y Latitud: Ubicación ')
st.write('Resolución: ')
st.write('Label: ')

st.markdown("---") # Linea divisoria

#-------------------------------------

#Análisis exploratorio
st.header('Análisis exploratorio')
st.markdown("---")

# Seleccion del dataset
st.subheader('Seleccione proyecto por estado (aprobados, desaprobados o en evaluación)')

st.markdown("##") # Linea en blanco

opcion_dataset = st.selectbox(
    '¿Qué dataset deseas visualizar?',
    ('Proyectos aprobados',
     'Proyectos desaprobados',
     'Proyectos en evaluacion')
    )
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

# Cruces de datos plot(x,y)

## Pie de frecuencia de actividad
t1 = '• Frecuencia de los proyectos '+estado+' según la clasificación ACTIVIDAD'
st.subheader(t1)
st.markdown("##")
df_actividad_freq = pd.DataFrame(df_visualizacion["ACTIVIDAD"].value_counts())
labels = df_actividad_freq.index.tolist()
sizes = df_actividad_freq["ACTIVIDAD"].tolist()
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
startangle=0)
#plt.title('Distribucion de datos segun ACTIVIDAD')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig1)
st.write('Figura 1. Gráfica pie de los proyectos con la frecuencia según el ACTIVIDAD de proyecto.')

## Pie de frecuencia de tipo
t2 = '• Frecuencia de los proyectos '+estado+' según la clasificación TIPO' 
st.subheader(t2)
st.markdown("##")
df_tipo_freq = pd.DataFrame(df_visualizacion["TIPO"].value_counts())
labels = df_tipo_freq.index.tolist()
sizes = df_tipo_freq["TIPO"].tolist()
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
startangle=0, textprops={'fontsize': 10})
#plt.title('Distribucion de datos segun TIPO')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig1)
st.write('Figura 2. Gráfica pie de los proyectos con la frecuencia según el TIPO de proyecto.')

## Cantidad de proyectos por año
t3 = '• Cantidad de proyectos '+estado+' presentados en diferentes años'
st.subheader(t3)
st.markdown("##")
df_anho_freq = pd.DataFrame(df_visualizacion["AÑO"].value_counts())
st.bar_chart(df_anho_freq)
st.write('Figura 3. Gráfica de barras de los proyectos clasificados por año de inicio.')

## Visualizacion de datos en el mapa
t4 = '• Proyectos '+estado+' localizados en un mapa interactivo mundial'
st.subheader(t4)
st.markdown("##") # Linea en blanco
df_visualizacion = df_visualizacion.rename(columns={'LATITUD':'lat', 'LONGITUD':'lon'})
st.map(df_visualizacion[['lat','lon']])
st.write('Figura 4. Ubicación de los proyectos',estado,'.')

## Tabla de datos (ver si se puede agregar widget de busqueda)
t5 = '• Se pueden visualizar '+str(len(df_visualizacion.index))+' proyectos '+estado+' :'
st.subheader(t5)
st.markdown("##") # Linea en blanco
st.dataframe(df_visualizacion)
st.markdown("##") # Linea en blanco
st.write('Tabla 1. Tabla de datos',estado,'en formato DataFrame.')

st.markdown("---") # Linea divisoria
#-----------------------------------------2DO----ANÁLISIS-----------------------------------------

# Análisis predictivo
st.header('Análisis predictivo')
st.subheader('Aplicación de los 5 pasos fundamentales para Machine Learning (Fig. 5.)')
st.markdown("---")

'''
Haciendo uso de los datasets de aprobados y desprobados, se entrenaron modelos de clasificación. Se utilizaron
las columnas de longitud, latitud, año y mes como X para predecir Y, que vendría a ser el Estado (categórico).

A continuación, usted podrá elegir el tipo de modelo (LR, SVM, RF, NN) que desee usar para obtener el archivo
donde se clasifica los proyectos que se encuentran en estado de evaluación como Aprobado o Desaprobado.
'''
st.markdown("##") # Linea en blanco
# Imagen de las fases del Machine Learning
image = Image.open('machinelearning.png')
st.image(image, caption='Figura 5. Pasos Fundamentales para el Machine Learning')
st.markdown("##") # Linea en blanco

# Selección del modelo para clasificar dataset de En evaluación
st.markdown("##") # Linea en blanco
st.write('Seleccionar tipo de modelo que se desea entrenar')
st.markdown("##") # Linea en blanco
opcion_modelo = st.selectbox(
     '¿Qué modelo deseas utilizar para el entrenamiento y posterior clasificación?',
     ('Logistic Regression','Support Vector Machine','Random Forest','Neural Network'))
modelo = None
if opcion_modelo == 'Logistic Regression':
    modelo = LR
elif opcion_modelo == 'Support Vector Machine':
    modelo = SVM
elif opcion_modelo == 'Random Forest':
    modelo = RF
elif opcion_modelo == 'Neural Network':
    modelo = NN
st.markdown("##") # Linea en blanco

# Nivel de predicción del modelo elegido
text2 = '• El score del modelo '+opcion_modelo+' es: '+str(round(modelo.score(X_test, y_test), 4))
st.subheader(text2)

st.markdown("##")
st.subheader("Se clasificaron los datos que tenían estado EN EVALUACIÓN utilizando el modelo",opcion_modelo)
st.write("Pulsar en el botón para descargar la información ya clasificada")

# Creación del excel con predicciones
X_gen = df_evaluacion.iloc[:,[10,11,14,15]]
y_gen = pd.DataFrame(modelo.predict(X_test))
df_evaluacion_gen = df_evaluacion
df_evaluacion_gen["ESTADO"] = y_gen

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

df_xlsx = to_excel(df_evaluacion_gen)

#Descarga del Excel con dataset nuevo con clasificaciones
st.markdown("##")
st.download_button(label='📥 Descargar aquí',
                                data=df_xlsx ,
                                file_name= 'df-evaluacion-clasificada-'+opcion_modelo.replace(' ','-').lower()+'.xlsx')

# Primero ordenar el codigo, colocando comentarios en #, no multilinea por que st lo reconoce
# Actualizar los textos del st, agregando tambien titulos, textos, espacios, lineas, donde se debe
# Cambiar los textos que figuren como lo del boton ,etc
# En base esto documentar primero en el markdow (readme) y en tu informe
