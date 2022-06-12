"""
git add . && git commit -m "" && git push
"""
# Importación de bibliotecas
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt

# Librerias y modulos de ML, DA
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Excel
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

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


# Preparacion de datos

# Selección de registros o entradas
"""
953
92
----
Correcion de datos (OK)
953
92

Aprobados: 953 = 800+153 (OK)
Desaprobados: 92 = 80+12 (OK)

Dataset con el 80% aprox. de la data original (OK)
Aprobados: 800 (obtenidos aleatoriamente) (OK)
Desaprobados: 80 (obtenidos aleatoriamente) (OK)
Desaprobados: 80x10 = 800 (obtenidos duplicando datos) (OK)
--------------------------------------------------
800+800 = 1600 datos (50%A y 50%D) Data balanceada

Modelo entrenado con 1600 -> modelo_clasificador

Para demostrar que es bueno, vas a seleccionar los 165 (153+12) datos
restantes con eso le demuestras a tu profe que esta bueno el modelo

Aplicas el modelo sobre la data que aun esta
pendiente de clasificacion (13 registros)
"""
# Dataframe aprobado se divide en 2: part1 (800) y parte2 (153)
df_aprobado_parte1  = df_aprobado.sample(n = 800, random_state=1)
df_aprobado_parte2 = df_aprobado.drop(df_aprobado_parte1.index)

# Reseteo de los indices de los datsets generados
df_aprobado_parte1.reset_index()
df_aprobado_parte2.reset_index()

#df_aprobado_parte1.info()
#df_aprobado_parte2.info()

# Dataframe desaprobado se divide en 2: part1 (80) y parte2 (12)
df_desaprobado_parte1  = df_desaprobado.sample(n = 80, random_state=1) # Esto para va train_test
df_desaprobado_parte2 = df_desaprobado.drop(df_desaprobado_parte1.index)

# Reseteo de los indices de los datsets generados
df_desaprobado_parte1.reset_index()
df_desaprobado_parte2.reset_index()

#df_desaprobado_parte1.info()
#df_desaprobado_parte2.info()

# Repetir 10 veces los 80 datos de la parte 1 del dataset desprobado = 10x80 = 800
df_desaprobado_parte1 =  pd.concat([df_desaprobado_parte1]*10)





# Creacion del dataframe para train y test del modelo
df_train =  pd.concat([df_aprobado_parte1, df_desaprobado_parte1])
df_test = pd.concat([df_aprobado_parte2, df_desaprobado_parte2])

df_train.reset_index()
df_test.reset_index()

df_train.info()



# Separacion de columnas X y columna y para a partir de los dataframe de train y tet
y_train = df_train.iloc[:,8]
X_train = df_train.iloc[:,[10,11,14,15]]

y_test = df_test.iloc[:,8]
X_test = df_test.iloc[:,[10,11,14,15]]







LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
#LR.predict(X_test)
print(round(LR.score(X_test,y_test), 4))

SVM = svm.SVC(decision_function_shape="ovo").fit(X_train, y_train)
#SVM.predict(X_test)
print(round(SVM.score(X_test, y_test), 4))

RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)
#RF.predict(X_test)
print(round(RF.score(X_test, y_test), 4))

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1).fit(X_train, y_train)
print(round(NN.score(X_test, y_test), 4))

"""
print("NN")
for i in range(7,11):
  for j in range(2,6):
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(i, j), random_state=1, max_iter=300).fit(X_train, y_train)
    acc = round(NN.score(X_test, y_test), 4)
    if acc > 0.8:
      print(i,j,acc)
"""  







#----------------------------------------
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

# Analisis predictivo
# Poner un texto que indique que haciendo uso de los dtasets de aprobados y desprobados
# se entrenaron varios modelos de clasificacion, utilizando las columnas
# lat y long (numericos), versus ESTADO (categorico)
# Se permitira al usuario seleccionar el tipo de modelo para mostrar el accuracy
# y en base a ellos se calificara a los datasets que aun estan evaluacion
# ...

st.write('Seleccionar tipo de modelo que se desea entrenar')
opcion_modelo = st.selectbox(
     '¿Qué modelo deseas utilizar pa el entranamiento?',
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

st.write('El score del modelo',opcion_modelo,'es:',round(modelo.score(X_test, y_test), 4))

st.write("Se clasificaron los datos quetenian estado PENDIENTE utiliznado el modelo",opcion_modelo)
st.write("Pulsar en el boton para descagr la informacion ya clasificada")

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
st.download_button(label='📥 Download Current Result',
                                data=df_xlsx ,
                                file_name= 'df-evaluacion-clasificada-'+opcion_modelo.replace(' ','-').lower()+'.xlsx')
