# native imports
import datetime

# common imports
import pandas as pd

# special imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.seasonal import seasonal_decompose

# front-end imports
import streamlit as st

# ===========================================
# Functions Data Retrieval
# ===========================================


@st.cache(allow_output_mutation=True)
def getDataFromCSV(file, sep, decimal) -> pd.DataFrame:
    dataFrame = pd.read_csv(file, sep, decimal,
                            encoding="UTF-8",
                            index_col=0,
                            low_memory=False)

    dataFrame.index = pd.to_datetime(
        dataFrame.index, format='%Y-%m-%d %H:%M:%S')
    dataFrame = dataFrame.sort_index(ascending=True)
    dataFrame = dataFrame.apply(pd.to_numeric, errors='coerce')
    return dataFrame

##########################
### App page beginning ###
##########################


def write(state):

    # Início da página Data Preparation
    st.title('Carregando e preparando os dados!')
    st.markdown(
        "O primeiro passo é fazermos a importação do arquivo **.csv** que contém os dados que deseja analisar.")

    st.info(
        """
        Atenção para a **formatação padrão** do arquivo **.csv**:
            \n* Delimitador = ","
            \n* Decimal = "."
            \n* Encoding = "UTF-8"
            \n* Datetime = "%Y-%m-%d %H:%M:%S"
            \n* Variável "Tempo" como sendo a coluna 0 do dataframe         
        """
    )

    input_csv_delimiter = st.selectbox("Delimitador", [",", ";", "|"])

    input_csv_decimal = st.selectbox("Decimal", [".", ","])

    uploaded_file = st.file_uploader(
        "Carregue aqui o seu arquivo csv",
        type="csv",
        key='uploaded_file')

    if uploaded_file:

        data_load_state = st.text('Carregando os dados...')

        df = getDataFromCSV(uploaded_file, input_csv_delimiter, input_csv_decimal).copy()

        data_load_state.text("Pronto! Dados carregados com sucesso!")
        st.markdown(
            "Para continuar, navegue pela aba **lateral esquerda** e selecione as opções.")

        if not df.empty:
            st.sidebar.title("Data Preparation")

            expanderFltTags = st.sidebar.beta_expander(
                label='Passo 1: Selecione as variáveis',
                expanded=False)

            state.dfTags = df.columns.values.tolist()

            state.fltTags = expanderFltTags.multiselect(
                label='Variáveis de Processo:',
                options=state.dfTags,
                default=state.fltTags,
                key='fltTagsPreparation')

            if len(state.fltTags) > 10:
                expanderFltTags.warning(
                    'Foram selecionadas mais de **10 TAGs**')

            dfRaw = df[state.fltTags]

            expanderFltDate = st.sidebar.beta_expander(
                label='Passo 2: Selecione as datas',
                expanded=False)

            startTimeDf = df.index[0]
            endTimeDf = df.index[-1]

            state.fltDateStart = expanderFltDate.date_input(
                label='Data de início:',
                value=state.fltDateStart or startTimeDf)
            state.fltTimeStart = expanderFltDate.time_input(
                label='Hora de início',
                value=state.fltTimeStart or datetime.time(0, 0, 0))

            state.fltDateEnd = expanderFltDate.date_input(
                label='Data de fim:',
                max_value=state.fltDateStart + datetime.timedelta(weeks=1),
                value=state.fltDateEnd or (state.fltDateStart + datetime.timedelta(weeks=1)))
            state.fltTimeEnd = expanderFltDate.time_input(
                label='Hora de fim',
                value=state.fltTimeEnd or datetime.time(endTimeDf.hour, endTimeDf.minute, endTimeDf.second))

            selStart = datetime.datetime.combine(
                state.fltDateStart, state.fltTimeStart)
            selEnd = datetime.datetime.combine(
                state.fltDateEnd, state.fltTimeEnd)

            expanderFltDate.warning(
                'O intervalo máximo é de 1 semana para um melhor desempenho')

            dfRawRange = dfRaw.loc[selStart:selEnd]
            state.dfRawRange = dfRawRange

            if state.fltTags != []:

                st.markdown("------------------------------------------")
                st.markdown(
                    "Perfeito, agora é só selecionar as opções abaixo para iniciar a exploração dos dados.")

                #################
                ### Dataframe ###
                #################

                showRawData = st.checkbox(
                    label='Visualizar os dados',
                    value=False,
                    key='showRawData')

                if (showRawData):
                    st.dataframe(data=dfRawRange)

                ############
                ### Info ###
                ############

                showInfo = st.checkbox(
                    label='Mostrar informaçoes do Dataframe (formato e missing values)', value=False, key='showInfo')
                if (showInfo):

                    dfInfo = pd.DataFrame()
                    dfInfo["Types"] = dfRawRange.dtypes
                    dfInfo["Missing Values"] = dfRawRange.isnull().sum()
                    dfInfo["Missing Values % "] = (
                        dfRawRange.isnull().sum()/len(dfRawRange)*100)
                    st.table(dfInfo)

                ################
                ### Cleaning ###
                ################

                state.execCleaning = st.checkbox(label='Fazer limpeza Automática dos dados', value=(
                    state.execCleaning or False), key='execCleaning')
                if (state.execCleaning):

                    methodCleaning = ["Interpolation", "Drop NaN"]

                    selectCleaning = st.selectbox(label='Selecione o método de limpeza',
                                                  options=methodCleaning,
                                                  key='selectCleaning')

                    if selectCleaning == "Drop NaN":

                        dfCleaned = dfRawRange.dropna(how="any")

                    elif selectCleaning == "Interpolation":

                        methodInterpolation = [
                            "linear", "nearest", "zero", "slinear", "quadratic", "cubic"]

                        selectInterpolation = st.selectbox(label='Selecione o método para a interpolação',
                                                           options=methodInterpolation,
                                                           key='selectInterpolation')

                        dfCleaned = dfRawRange.interpolate(
                            method=selectInterpolation, inplace=False)

                    st.text("Informações do Dataframe após a limpeza.")

                    state.dfRawRange = dfCleaned

                    dfInfo = pd.DataFrame()
                    dfInfo["Types"] = state.dfRawRange.dtypes
                    dfInfo["Missing Values"] = state.dfRawRange.isnull().sum()
                    dfInfo["Missing Values % "] = (
                        state.dfRawRange.isnull().sum()/len(state.dfRawRange)*100)

                    st.table(dfInfo)
                    st.markdown(" ")
                    st.markdown("**Atenção**: Enquanto esta opção *'Fazer limpeza Automática dos dados'*\
                                estiver selecionada, os dados considerados serão *'Dataframe após a limpeza.'*")
                    st.markdown(" ")

                ################
                ### Describe ###
                ################

                showDescribe = st.checkbox(
                    label='Mostrar estatística descritiva', value=False, key='showDescribe')
                if (showDescribe):
                    st.table(dfRawRange.describe().transpose())

                ##########################
                ### Análise Individual ###
                ##########################

                showInfoVar = st.checkbox(
                    label="Análisar graficamente cada variável (tendência, sazonalidade, histogram e boxplot",
                    value=False,
                    key='showInfoVar')
                # Lógica IF para fazer aparecer a tendencia e sazonalidade quando a variável não possuir
                # valores nulos. Caso tenha valores nulos é plotado apenas a variavel
                if (showInfoVar):

                    st.write("Selecione a váriavel a ser analisada")
                    fltPlot = st.selectbox(
                        label='', options=state.fltTags, key='fltPlot')

                    if state.dfRawRange[fltPlot].isnull().sum() > 0:

                        figDecompose = go.Figure()

                        figDecompose.add_trace(go.Scatter(x=state.dfRawRange.index,
                                                          y=state.dfRawRange[fltPlot],
                                                          name=fltPlot))

                    else:
                        st.write(
                            "Análise de decomposição da série temporal (tendência, sazonalidade e ruído)")
                        periodDecompose = st.number_input(
                            "Informe o período de amostragem da série temporal",
                            min_value=2,
                            max_value=360,
                            value=2,
                            step=1,
                            format="%i",
                            key="periodDecompose")

                        serieDecompose = seasonal_decompose(
                            state.dfRawRange[fltPlot], model='additive', period=periodDecompose)

                        figDecompose = make_subplots(
                            rows=4, cols=1, shared_xaxes=True)

                        figDecompose.add_trace(go.Scatter(x=state.dfRawRange.index,
                                                          y=state.dfRawRange[fltPlot],
                                                          name=fltPlot),
                                               row=1, col=1)

                        figDecompose.add_trace(go.Scatter(x=state.dfRawRange.index,
                                                          y=serieDecompose.trend,
                                                          name="Tendência"),
                                               row=2, col=1)

                        figDecompose.add_trace(go.Scatter(x=state.dfRawRange.index,
                                                          y=serieDecompose.seasonal,
                                                          name="Sazonalidade"),
                                               row=3, col=1)

                        figDecompose.add_trace(go.Scatter(x=state.dfRawRange.index,
                                                          y=serieDecompose.resid,
                                                          name="Resíduos"),
                                               row=4, col=1)

                        figDecompose.update_layout(
                            xaxis2_rangeslider_visible=False,
                            xaxis2_rangeslider_thickness=0.1)

                    figDecompose.update_layout(
                        xaxis=dict(
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=30,
                                         label="30min",
                                         step="minute",
                                         stepmode="backward"),
                                    dict(count=1,
                                         label="1h",
                                         step="hour",
                                         stepmode="backward"),
                                    dict(count=1,
                                         label="1d",
                                         step="day",
                                         stepmode="backward"),
                                    dict(count=7,
                                         label="1w",
                                         step="day",
                                         stepmode="backward"),
                                    dict(count=1,
                                         label="1m",
                                         step="month",
                                         stepmode="backward"),
                                    dict(step="all")
                                ])
                            )
                        )
                    )
                    st.plotly_chart(figDecompose)

                    st.info(
                        "Para cada gráfico de tendência, sazonalidade e ruído, se ocorrerem valores maiores que zero,\
                        significa que a série pode ter algum desses fatores e, nesse caso,\
                        deve-se avaliar mais a fundo técnicas para compensar tais fatores.",
                    )

                    st.write("Histograma e Boxplot")
                    figHist = px.histogram(
                        state.dfRawRange, x=fltPlot, marginal="box")
                    st.plotly_chart(figHist)
