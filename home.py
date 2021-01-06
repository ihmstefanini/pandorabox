# native imports

# common imports
import numpy as np
import pandas as pd
from PIL import Image
image1 = Image.open('IHM Pandora.jpeg')

# special imports

# front-end imports
import streamlit as st

# ===========================================
# Functions Data Retrieval
# ===========================================
def write(state):


    #st.sidebar.header("About")
    #st.sidebar.info("Essa é uma versão de demo do IHM Pandora Box. Maiores informações, entre em contato pelo email: eduardo.magalhaes@ihm.com.br")


    # ===========================================
    # Body
    # ===========================================
    
    #st.title('Qual problema esse app resolve?')
    #st.subheader("Automatiza a sincronização entre variáveis de processo e de qualidade, entregando data sets prontos para a análise")
    st.markdown('')
    st.markdown('')

    st.title('Bem vindo o IHM Pandora Box!')
    #st.image(image2,use_column_width=True)
    
    st.markdown('')
    #st.markdown('O primeiro aplicativo criado para calcular o **potencial preditivo** dos problemas de dados da indústria!') 
    st.subheader("O aplicativo que avalia o potencial preditivo do seu processo industrial em minutos")
    st.markdown(' ')
    st.markdown(' ')
    #st.image(image, caption='Arquitetura',use_column_width=True)
    st.subheader('**Por quê o Pandora Box?**')
    st.markdown('Se você precisa fazer a predição de uma variável de processo ou qualidade e não sabe se é viável fazer um modelo preditivo para isso, então o Pandora Box é o app certo para responder essa pergunta.')
    #st.subheader("Automatiza a sincronização entre variáveis de processo e de qualidade, entregando data sets prontos para a análise")
    st.markdown('')   
    st.subheader('**Quais são os benefícios do Pandora?**')
    #st.subheader("Evita tempo gasto com limpeza, tratamento e organização dos dados, permitindo que vocês e seu time se concentrem na parte nobre da análise exploratória e modelagem dos dados. ")
    st.markdown('''
    
    >1. Análise rápida do potencial preditivo do seu problema, de forma simples, rápida e descomplicada;

    >2. Análise do atraso de tempo entre variáveis, permitindo avaliar o sincronismo dos dados no tempo;

    >3. Cálculos de correlação que levam em conta a dinâmica não linear do processo
    
    >4. Testes de causalidade e significância estatística. 

    ''')
    #>3. Agilidade para identificar a correlação entre dezenas de variáveis simultaneamente. Algoritmos tradicionais levam horas para correlacionar centenas de variáveis que, usando Pandora Box, pode ser feito em minutos.
    
    st.markdown('')

    st.subheader('**Como funciona?**')
    st.markdown("Basta carregar uma pequena base de dados no formato *.CSV e ir navegando no aplicativo que avalia que o Pandora já vai calculando tudo automaticamente. No fim, o app **estima o potencial preditivo do seu problema**, permitindo uma avaliação mais criterio da viabilidade de se modelar preditivamente o problema.")
    st.image(image1,use_column_width=True)#, caption='Arquitetura')
    st.markdown('')
    
    st.markdown('')
    st.subheader('**Bora experimentar?**')
    st.markdown("Na aba de navegação à esquerda, selecione **Data Preparation**")
    st.markdown('')
    st.markdown('')
    # ===========================================
    # Raw
    # ===========================================