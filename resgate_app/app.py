import joblib
import pandas as pd
import streamlit as st

from transformadores import ExtrairTitulo  # necessário para carregar o pickle

st.set_page_config(page_title='Resgate SpaceTec-Info EAJ', page_icon='🚀')


@st.cache_resource

BASE_DIR = Path(__file__).resolve().parent
CAMINHO_MODELO = BASE_DIR / "modelo_resgate.pkl"

def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)


modelo = carregar_modelo()

st.title('🚀 Previsão de resgate - Estação SpaceTec-Info EAJ')
st.write('Informe os dados do tripulante para prever se ele foi **resgatado** '
         'ou ficou **perdido na órbita**.')

with st.form('formulario'):
    col1, col2 = st.columns(2)

    with col1:
        titulo = st.selectbox('Título', ['Passag.', 'Eng.', 'Ten.', 'Dra.', 'Cap.'])
        nome = st.text_input('Nome', value='Tripulante_0')
        modulo = st.selectbox('Módulo da estação',
                              ['Laboratorio', 'Engenharia', 'Habitacao', 'Comando'])
        classe = st.selectbox('Classe do bilhete', [1, 2, 3], index=2)
        traje = st.radio('Traje pressurizado', ['Sim', 'Nao'], horizontal=True)

    with col2:
        idade_conhecida = st.checkbox('Idade conhecida', value=True)
        idade = st.number_input('Idade', min_value=0.0, max_value=120.0,
                                value=30.0, step=0.5, disabled=not idade_conhecida)
        parentes = st.number_input('Parentes a bordo', min_value=0, max_value=10, value=0)
        horas = st.number_input('Horas de treinamento', min_value=0.0,
                                max_value=500.0, value=40.0, step=0.5)
        tarifa = st.number_input('Tarifa (créditos)', min_value=0.0,
                                 max_value=300.0, value=80.0, step=0.5)

    enviar = st.form_submit_button('Prever')

if enviar:
    entrada = pd.DataFrame([{
        'modulo_estacao': modulo,
        'classe_bilhete': classe,
        'nome_tripulante': f'{titulo} {nome}',
        'idade': idade if idade_conhecida else float('nan'),  # o pipeline imputa a mediana
        'parentes_a_bordo': parentes,
        'horas_treinamento': horas,
        'traje_pressurizado': traje,
        'tarifa_creditos': tarifa,
    }])

    pred = int(modelo.predict(entrada)[0])
    prob = float(modelo.predict_proba(entrada)[0][1])

    if pred == 1:
        st.success(f'✅ Resgatado (probabilidade de resgate: {prob:.1%})')
    else:
        st.error(f'⚠️ Perdido na órbita (probabilidade de resgate: {prob:.1%})')

    with st.expander('Dados enviados ao modelo'):
        st.dataframe(entrada)
