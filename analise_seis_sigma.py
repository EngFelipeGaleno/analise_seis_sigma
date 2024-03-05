# !pip install streamlit
# Commented out IPython magic to ensure Python compatibility.
# %reset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dateutil.parser
import seaborn as sns
import sklearn
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")

df = pd.read_excel('Produção_CKS.xlsm', sheet_name='Variabilidade', usecols=list(range(5)))

df['mes'] = pd.to_datetime(df['Data']).dt.month
df['ano'] = pd.to_datetime(df['Data']).dt.year

def clima(mes):
    if mes == 1 or mes == 2 or mes == 3 or mes == 4:
        return 'chove'
    elif mes == 11 or mes == 12:
        return 'intermediario'
    elif mes == 6 or mes == 7 or mes == 8 or mes == 9 or mes == 10:
        return 'seco'
    else:
        return 'chove'

# Aplicando a função usando apply e lambda
df['clima'] = df['mes'].apply(lambda x: clima(x))



df_cv = df.groupby(['ano', 'mes','Sistema produtivo'])['Produção_ajuste'].agg(['mean', 'std']).round(2)
df_cv.columns = ['Média', 'Desvio_Padrão']
df_cv['Coeficiente_var'] = round(df_cv['Desvio_Padrão'] / df_cv['Média'], 4)*100
df_cv = df_cv.reset_index()

df_cv_acumulado = df_cv.groupby(['Sistema produtivo'])['Média'].agg(['mean', 'std']).round(2)
df_cv_acumulado.columns = ['Média', 'Desvio_Padrão']
df_cv_acumulado['Coeficiente_var'] = round(df_cv_acumulado['Desvio_Padrão'] / df_cv_acumulado['Média'], 4)*100
df_cv_acumulado = df_cv_acumulado.reset_index()

df_cv_total = df_cv.groupby(['ano', 'mes'])['Média'].agg(['sum', 'std']).round(2)
df_cv_total.columns = ['Média', 'Desvio_Padrão']
df_cv_total['Coeficiente_var'] = round(df_cv_total['Desvio_Padrão'] / df_cv_total['Média'], 4)*100
df_cv_total = df_cv_total.reset_index()

st.title('Dashboard de Produção')

st.header('Média Mensal de Produção por Sistema Produtivo')
st.dataframe(df_cv.style.background_gradient(cmap='Greens'))

st.header('Média Mensal de Produção Total')
st.dataframe(df_cv_total.style.background_gradient(cmap='Greens'))

st.header('Média Anual de Produção por Sistema Produtivo')
st.dataframe(df_cv_acumulado.style.background_gradient(cmap='Greens'))

st.header('Gráfico de Média Mensal de Produção por Sistema Produtivo')
fig = px.bar(df_cv, x='mes', y='Média', color='Sistema produtivo', barmode='group')
st.plotly_chart(fig)

st.header('Gráfico de Média Mensal de Produção Total')
fig = px.bar(df_cv, x='mes', y='Média', color='Sistema produtivo', barmode='group')
st.plotly_chart(fig)

st.header('Gráfico de Média Anual de Produção por Sistema Produtivo')
fig = px.bar(df_cv_acumulado, x='Sistema produtivo', y='Média', color='Sistema produtivo', barmode='group')
st.plotly_chart(fig)
