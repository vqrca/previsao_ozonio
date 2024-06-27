import streamlit as st
import pandas as pd
import json
from prophet.serialize import model_from_json
from prophet.plot import plot_plotly

# Carregar o modelo Prophet salvo
def load_model():
    with open('modelo_O3_prophet.json', 'r') as file_in:
        modelo = model_from_json(json.load(file_in))
        return modelo

modelo = load_model()

# Layout do Streamlit
st.title('Previsão de Níveis de Ozônio (O3) Utilizando a Biblioteca Prophet')

# Adicionando uma área de texto para a descrição do projeto
st.caption('''Este projeto utiliza a biblioteca Prophet para prever os níveis de ozônio em ug/m3. O modelo
           criado foi treinado com dados até o dia 05/05/2023 e possui um erro de previsão (RMSE - Erro Quadrático Médio) igual a 17.43 nos dados de teste. 
           O usuário pode inserir o número de dias para os quais deseja a previsão, e o modelo gerará um gráfico
           interativo contendo as estimativas baseadas em dados históricos de concentração de O3.
           Além disso, uma tabela será exibida com os valores estimados para cada dia.''')

st.subheader('Insira o número de dias para previsão:')

# Recebendo o número de dias do usuário
dias = st.number_input('', min_value=1, value=1, step=1)

# Gerenciamento de estado
if 'previsao_feita' not in st.session_state:
    st.session_state.previsao_feita = False
    st.session_state.tabela_previsao = pd.DataFrame()

if st.button('Prever') or st.session_state.previsao_feita:
    st.session_state.previsao_feita = True
    # Criando dataframe futuro e realizando previsão
    futuro = modelo.make_future_dataframe(periods=dias, freq='D')
    previsao = modelo.predict(futuro)

    # Plotando o gráfico interativo
    st.write('Gráfico da previsão')
    fig = plot_plotly(modelo, previsao)

    # Definindo o fundo do gráfico como branco
    fig.update_layout({
    'plot_bgcolor': 'rgba(255, 255, 255, 1)',  # Define o fundo da área do gráfico como branco
    'paper_bgcolor': 'rgba(255, 255, 255, 1)',  # Define o fundo externo ao gráfico como branco
})

# Ajustando a cor do texto para preto para melhor visibilidade
    fig.update_layout(
    title={'text': "Previsão de Ozônio", 'font': {'color': 'black'}},
    xaxis={'title': 'Data', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}},
    yaxis={'title': 'Nível de Ozônio (O3 μg/m3)', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}}
    )

    st.plotly_chart(fig)

 # Preparando a tabela de previsões
    st.session_state.tabela_previsao = previsao[['ds', 'yhat']].tail(dias)
    st.session_state.tabela_previsao.columns = ['Data (Dia/Mês/Ano)', 'O3 (μg/m3)']
    st.session_state.tabela_previsao['Data (Dia/Mês/Ano)'] = st.session_state.tabela_previsao['Data (Dia/Mês/Ano)'].dt.strftime('%d-%m-%Y')
    st.session_state.tabela_previsao['O3 (μg/m3)'] = st.session_state.tabela_previsao['O3 (μg/m3)'].round(2)
    st.session_state.tabela_previsao.reset_index(drop=True, inplace=True)

if st.session_state.previsao_feita:

    # Exibindo e permitindo o download da tabela
    st.write("Tabela contendo as previsões de ozônio (μg/m3) para os próximos {} dias:".format(dias))
    st.dataframe(st.session_state.tabela_previsao, height=300)
    
    # Convertendo DataFrame para CSV e criando link de download
    csv = st.session_state.tabela_previsao.to_csv(index=False)
    st.download_button(label="Baixar tabela como CSV", data=csv, file_name='previsao_ozonio.csv', mime='text/csv')
