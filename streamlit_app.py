"""
git add . && git commit -m "" && git push
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
st.title('Test')
fecha_inicio = st.slider("Ver informacion registrada en:",
    value = datetime(2020,1,1,9,30),
    format = "DD/MM/YY - hh:mm")
st.write("Fecha seleccionada:", fecha_inicio)
