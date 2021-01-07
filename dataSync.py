# common imports
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# special imports
from minepy import MINE
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

# front-end imports
import streamlit as st

# ===========================================
# Functions Data Retrieval
# ===========================================

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def getBestLags(df, tags, target, shiftFrom, shiftTo, shiftStep):
    
    barProgresso = st.progress(0)
    progressoAtual = 0
    progressoTotal  = len(tags) * len(range(shiftFrom, shiftTo, shiftStep))
    
    resultPearson = {}
    resultMic = {}
    resultSpearman = {}
    resultKendall = {}
    
    for tag in tags:
        
        vetorPearson = {}
        vetorMic = {}
        vetorSpearman = {}
        vetorKendall = {}
        mine = MINE(alpha=0.6, c=15, est="mic_e") 
                
        for lag in range(shiftFrom, shiftTo, shiftStep):
            
            tempDf = pd.concat([df[tag].shift(lag, freq='min'), df[target]], axis=1)
            tempDf.dropna(inplace = True)
                 
            vetorPearson[lag] = tempDf.corr(method='pearson').iloc[0][1]                                        
            vetorSpearman[lag] = tempDf.corr(method='spearman').iloc[0][1]                                                  
            vetorKendall[lag] = tempDf.corr(method='kendall').iloc[0][1]
            
            mine.compute_score(tempDf[tag], tempDf[target]) 
            vetorMic[lag] = mine.mic()

            progressoAtual += 1
            barProgresso.progress(int(progressoAtual*100/progressoTotal))
                     
        resultPearson[tag] = pd.DataFrame.from_dict(vetorPearson, orient='index', columns=[tag])
        resultMic[tag] = pd.DataFrame.from_dict(vetorMic, orient='index', columns=[tag])
        resultSpearman[tag] = pd.DataFrame.from_dict(vetorSpearman, orient='index', columns=[tag])
        resultKendall[tag] = pd.DataFrame.from_dict(vetorKendall, orient='index', columns=[tag])
    
    return resultPearson, resultMic, resultSpearman, resultKendall

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def getMaxCorr(dicRawBestLag, fltTags): 
    
    dfMaxCorr = []
    dfMinCorr = []
    dfIdxMinCorr = []
    dfIdxMaxCorr = []

    for k in dicRawBestLag.keys():

        dfMaxCorr.extend(dicRawBestLag.get(str(k)).max())
        dfMinCorr.extend(dicRawBestLag.get(str(k)).min())
        dfIdxMaxCorr.extend(dicRawBestLag.get(str(k)).idxmax())
        dfIdxMinCorr.extend(dicRawBestLag.get(str(k)).idxmin())

    maximumcorr = []
    maximumcorridx = []
    
    for v in range(len(fltTags)):
        
        if (abs(dfMinCorr[v]) > abs(dfMaxCorr[v])):
                maximumcorr.append(dfMinCorr[v])
                maximumcorridx.append(dfIdxMinCorr[v])

        else:
            maximumcorr.append(dfMaxCorr[v])
            maximumcorridx.append(dfIdxMaxCorr[v])
            
    return maximumcorr, maximumcorridx

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calcCorr(df, tags, target):
    
    dfCorrPearson = df[tags].corr(method='pearson')
    pearson = dfCorrPearson[dfCorrPearson.index == target] 

    dfCorrSpearman = df[tags].corr(method='spearman')
    spearman = dfCorrSpearman[dfCorrSpearman.index == target]
    
    dfCorrKendall = df[tags].corr(method='kendall')
    kendall = dfCorrKendall[dfCorrKendall.index == target]
    
    mic = pd.DataFrame()                
    mine = MINE(alpha=0.6, c=15, est="mic_e") 
    
    for  tag in tags:

        mine.compute_score(df[tag], df[target]) 
        mic.loc[0, tag] = mine.mic()        
    
    dfCorr = pd.concat([pearson, spearman, kendall])
    dfCorr.index = ["Pearson", "Spearman", "Kendall"]
    dfCorr.drop([target], axis=1, inplace=True)

    return dfCorr.transpose()

##########################
### App page beginning ###
##########################

def write(state):

    if type(state.dfRawRange) == type(None):
        st.warning("Realize a importação do arquivo CSV contendo os dados em **'Data Preparation'**.")
        
    elif state.dfRawRange.empty:

        st.warning("Realize os passos 1 e 2 do **'Data Preparation'**.")
        
    else:

        #st.markdown("Pronto! Dados carregados com sucesso!")        
        st.sidebar.title("Data Syncronization")
        
        # ===========================================
        # Body
        # ===========================================
        
        st.title('Obtendo os datasets sincronizados')
        st.markdown('A última etapa para finalmente obter os dados **sincronizados** é utilizar o cálculo de correlação para encontrar a correta **defasagem (lag)** entre a variável de processo e de qualidade.')
        st.subheader("")

        # ===========================================
        # Raw
        # ===========================================
        
        if type(state.fltTarget) == type(None):
            st.warning('**Atenção** - Antes de iniciar essa etapa, navegue pela aba **Data Correlation** e **selecione** uma variável alvo no passo 1')
            
        else:

            st.markdown('1. Primeiro vamos recapitular o que descobrimos na etapa anterior. Vejamos então a **matriz de correlação** das variáveis selecionadas em relação ao targert **{}**.'\
                .format(state.fltTarget))
            
            if len(state.fltTags) > 0:

                dfRawCorr = calcCorr(state.dfRawRange, state.fltTags, state.fltTarget)
                st.dataframe(dfRawCorr)

                st.info("Os resultados do cálculo de correlação são a base para o cálculo de defasagem (lag) que iremos realizar no Passo 2.")
                st.subheader('')
        
                st.markdown('2. Vamos agora utilizar o algoritmo de varredura para encontrar o melhor valor de defasagem entre a(s) variável(is) de processo e qualidade.')
                st.markdown('Entretanto, o algoritmo de varredura precisa de saber o **tamanho da janela de varredura**. Nesse caso, selecione o tamanho da janela (escala em minutos):')
                st.markdown('')
                     
                parSearchLimits = st.slider(label='Selecione a janela de busca (minutos)', min_value=0, max_value=360, value=(0, 360), step=5)
                
                fltTags = state.fltTags.copy()
                fltTags.remove(state.fltTarget)
                
                btnBestLag = st.button("Calcular BestLag", key="btnBestLag")
                st.warning("**Nota**: quanto maior a janela de varredura, maior o tempo de execução dos cálculos.")
                
                if btnBestLag:
                    
                    st.warning("Cálculo do bestlag em execução!")
                    
                    state.dicRawBestLagPearson, state.dicRawBestLagMic, state.dicRawBestLagSpearman, state.dicRawBestLagKendall\
                        = getBestLags(df=state.dfRawRange, tags=fltTags, target=state.fltTarget, shiftFrom=parSearchLimits[0], shiftTo=parSearchLimits[1], shiftStep=5)
                   
                    #Busca os valores máx e min e seus respectivos indices na função de busca da correlacao          
                    state.maximumPearson, state.maximumidxPearson = getMaxCorr(state.dicRawBestLagPearson, fltTags)
                    state.maximumSpearman, state.maximumidxSpearman = getMaxCorr(state.dicRawBestLagSpearman, fltTags)
                    state.maximumKendall, state.maximumidxKendall = getMaxCorr(state.dicRawBestLagKendall, fltTags)
                    state.maximumMic, state.maximumidxMic = getMaxCorr(state.dicRawBestLagMic, fltTags)
                    

                    st.success("Finalizado!")
                        
                fltRawBestLagCurve = st.selectbox(label='Process Variable:', options=fltTags, key='fltRawBestLagCurve')
                if len(fltRawBestLagCurve) > 0:
                    
                    for k in range(len(fltTags)):
                        if (fltTags[k] == fltRawBestLagCurve):
                            indice_variavel = k
                    
                    if (state.dicRawBestLagPearson != None) & (state.dicRawBestLagSpearman != None) & (state.dicRawBestLagKendall != None) & (state.dicRawBestLagMic != None) :

                        dict_best_lags = {}   
                        dict_best_lags['pearson'] =  state.maximumidxPearson[indice_variavel]                
                        #dict_best_lags['mic'] =  state.maximumidxMic[indice_variavel]           
                        dict_best_lags['spearman'] =  state.maximumidxSpearman[indice_variavel]           
                        dict_best_lags['kendall'] =  state.maximumidxKendall[indice_variavel]      

                        #Mostrar na tela o valor e o índice (tempo) em que a máxima correlação foi encontrada, o que significa o melhor valor do best lag
                        st.info("**Pearson:** o maior valor de correlação entre o target **{}** e a variável **{}** foi de **{:.2f}** onde o valor do tempo de atraso (best lag) é igual a **{}** minutos."\
                            .format(state.fltTarget, fltRawBestLagCurve, state.maximumPearson[indice_variavel] ,state.maximumidxPearson[indice_variavel]))
                        
                        st.info("**Spearman:** o maior valor de correlação entre o target **{}** e a variável **{}** foi de **{:.2f}** onde o valor do tempo de atraso (best lag) é igual a **{}** minutos."\
                            .format(state.fltTarget, fltRawBestLagCurve, state.maximumSpearman[indice_variavel] ,state.maximumidxSpearman[indice_variavel]))

                        st.info("**Kendall:** o maior valor de correlação entre o target **{}** e a variável **{}** foi de **{:.2f}** onde o valor do tempo de atraso (best lag) é igual a **{}** minutos."\
                            .format(state.fltTarget, fltRawBestLagCurve, state.maximumKendall[indice_variavel] ,state.maximumidxKendall[indice_variavel]))
                        
                        #st.info("**MIC:** o maior valor de correlação entre o target **{}** e a variável **{}** foi de **{:.2f}** onde o valor do tempo de atraso (best lag) é igual a **{}** minutos."\
                        #    .format(state.fltTarget, fltRawBestLagCurve, state.maximumMic[indice_variavel] ,state.maximumidxMic[indice_variavel]))
                        
                        #########################################
                        
                        figBestLag, annotations = plt.subplots() 
                        annotations.plot(state.dicRawBestLagPearson[fltRawBestLagCurve], lw = 1, color='r',label="Pearson") 
                        annotations.scatter(x=state.maximumidxPearson[indice_variavel], y=state.maximumPearson[indice_variavel], color="red")
                        
                        annotations.plot(state.dicRawBestLagSpearman[fltRawBestLagCurve], lw = 1, color='orange',label="Spearman") 
                        annotations.scatter(x=state.maximumidxSpearman[indice_variavel], y=state.maximumSpearman[indice_variavel], color="orange")

                        annotations.plot(state.dicRawBestLagKendall[fltRawBestLagCurve], lw = 1, color='k',label="Kendall") 
                        annotations.scatter(x=state.maximumidxKendall[indice_variavel], y=state.maximumKendall[indice_variavel], color="k")
                        
                        #annotations.plot(state.dicRawBestLagMic[fltRawBestLagCurve], lw = 1, color='g',label="MIC") 
                        #annotations.scatter(x=state.maximumidxMic[indice_variavel], y=state.maximumMic[indice_variavel], color="g")
 
                        annotations.set_title('Métodos de Correlação na Janela de Varredura \n {} x {}'.format(state.fltTarget, fltRawBestLagCurve ))
                        annotations.set_ylabel('Scores')
                        annotations.set_xlabel('Deslocamento (min)')
                        annotations.legend(bbox_to_anchor=(1.3, 1))
                        annotations.set_ylim(-1.1, 1.1)
                        
                        st.pyplot(figBestLag)

                        ############################################################
                        ############# Teste de hipotese de causalidade #############
                        ############################################################
                        # Variaveis independentes: state.fltTags
                        # Variavel dependente: state.fltTarget
                        # df = state.dfRawRange
                        dependent_variable = state.fltTarget
                        independent_variables = set(state.fltTags) - set(dependent_variable)

                        #for i in range(0, len(independent_variables)):
                        list_columns = []
                        list_columns.append(state.fltTarget)
                        list_columns.append(str(fltRawBestLagCurve))

                        k_v_exchanged = {}

                        for key, value in dict_best_lags.items():
                            if value not in k_v_exchanged:
                                k_v_exchanged[value] = [key]
                            else:
                                k_v_exchanged[value].append(key)

                        list_lags = list(k_v_exchanged.keys())

                        #list_correlation_methods = list(dict_best_lags.keys())
                        gc_res = {}
                        p_value_ssr_ftest = {}
                        p_value_ssr_chi2test = {}
                        p_value_lrtest = {}
                        p_value_params_ftest = {}

                        for lag in list_lags:
                            counter = 0
                            if(lag>0):
                                gc_res[lag] = grangercausalitytests(state.dfRawRange[list_columns], [lag])
                            else:
                                gc_res[lag] = None

                            if (gc_res[lag] is not None):
                                p_value_ssr_ftest[lag] = gc_res[lag][lag][0]['ssr_ftest'][1]
                                p_value_ssr_chi2test[lag] = gc_res[lag][lag][0]['ssr_chi2test'][1]
                                p_value_lrtest[lag] = gc_res[lag][lag][0]['lrtest'][1]
                                p_value_params_ftest[lag] = gc_res[lag][lag][0]['params_ftest'][1]

                            list_methods = k_v_exchanged[lag]    

                            if(lag>0):
                                if(p_value_ssr_ftest[lag]<0.05):
                                    counter = counter + 1
                                if(p_value_ssr_chi2test[lag]<0.05):
                                    counter = counter + 1
                                if(p_value_lrtest[lag]<0.05):
                                    counter = counter + 1
                                if(p_value_params_ftest[lag]<0.05):
                                    counter = counter + 1
                            
                            if(len(list_methods)>1):   
                                methods = ''                     
                                for method in list_methods:
                                    methods = methods + method + ', '
                                methods = methods[:-2]
                                if(counter>0):
                                    st.markdown("Resultado dos **testes de hipótese** para o **best lag** encontrado pelos métodos **" + methods + "**:")
                                    st.info("Valores passados (" + str(lag) + " minutos) de " + list_columns[1] + " tem um efeito estatisticamente significante nos valores atuais de " + list_columns[0] + ". Passou em " + str(np.round(((counter/4)*100), 2)) + " % dos testes de hipótese.")
                                else:
                                    st.markdown("Resultado dos **testes de hipótese** para o **best lag** encontrado pelos métodos " + methods + ":")
                                    st.info("Valores passados (" + str(lag) + " minutos) de " + list_columns[1] + " não tem um efeito estatisticamente significante nos valores atuais de " + list_columns[0] + ". Passou em " + str(np.round(((counter/4)*100), 2)) + " % dos testes de hipótese.")
                   
                            else:                           
                                if(counter>0):
                                    st.markdown("Resultado dos **testes de hipótese** para o **best lag** encontrado pelo método **" + k_v_exchanged[lag][0] + "**:")
                                    st.info("Valores passados (" + str(lag) + " minutos) de " + list_columns[1] + " tem um efeito estatisticamente significante nos valores atuais de " + list_columns[0] + ". Passou em " + str(np.round(((counter/4)*100), 2)) + " % dos testes de hipótese.")
                                else:
                                    st.markdown("Resultado dos **testes de hipótese** para o **best lag** encontrado pelo método **" + k_v_exchanged[lag][0] + "**:")
                                    st.info("Valores passados (" + str(lag) + " minutos) de " + list_columns[1] + " não tem um efeito estatisticamente significante nos valores atuais de " + list_columns[0] + ". Passou em " + str(np.round(((counter/4)*100), 2)) + " % dos testes de hipótese.")

                        state.fim_etapa2 = 1
                    
            if ((len(state.fltTags) > 0)):
                st.markdown('3. Vamos agora **reconstruir o dataset** considerando o atraso de tempo encontrado, de forma que os dados ficarão **sincronizados**, facilitando a sua aplicação para os algoritmos de modelagem.')
                #st.markdown('Entretanto, o algoritmo de varredura precisa de saber o **tamanho da janela de varredura**. Nesse caso, selecione o tamanho da janela (escala em minutos):')
                st.markdown('')
                if st.checkbox("Clique aqui para ver os datasets sincronizados"):
                    st.write("Esse é o dataset original")
                    st.dataframe(state.dfRawRange)
                    st.markdown('')
                    
                    methodLag = ["Pearson", "Spearman", "Kendall"]#, "Mic"]
                    
                    fltMethodLag = st.selectbox(label='Metódo do lag: ', options=methodLag, key='fltMethodLag')
                    st.write("Esse é o dataset reconstruído. Perceba que há um **deslocamento** no valor do time stamp na primeira linha, devido a consideração do atraso **(best lag)** calculado.")
                    
                    state.dfRawRangeShifted = state.dfRawRange.copy()
                    
                    for tag in fltTags:
                        
                        indexTag = fltTags.index(tag)
                        
                        if fltMethodLag == "Pearson":
                            if np.isnan(state.maximumidxPearson[indexTag]):
                                st.warning("Não foi possível realizar a correlação de pearson entre **{}** e **{}**, portanto não será realizado o descolamento de **{}**."\
                                    .format(tag, state.fltTarget, tag))
                                state.dfRawRangeShifted[tag] = state.dfRawRangeShifted[tag].shift(0, freq='min')
                            else:
                                state.dfRawRangeShifted[tag] = state.dfRawRangeShifted[tag].shift(state.maximumidxPearson[indexTag], freq='min')
            
                        elif fltMethodLag == "Spearman":
                            if np.isnan(state.maximumidxSpearman[indexTag]):
                                st.warning("Não foi possível realizar a correlação de spearman entre **{}** e **{}**, portanto não será realizado o descolamento de **{}**."\
                                    .format(tag, state.fltTarget, tag))
                                state.dfRawRangeShifted[tag] = state.dfRawRangeShifted[tag].shift(0, freq='min')
                            else:
                                state.dfRawRangeShifted[tag] = state.dfRawRangeShifted[tag].shift(state.maximumidxSpearman[indexTag], freq='min')
                        
                        elif fltMethodLag ==  "Kendall":
                            if np.isnan(state.maximumidxKendall[indexTag]):
                                st.warning("Não foi possível realizar a correlação de kendall entre **{}** e **{}**, portanto não será realizado o descolamento de **{}**."\
                                    .format(tag, state.fltTarget, tag))
                                state.dfRawRangeShifted[tag] = state.dfRawRangeShifted[tag].shift(0, freq='min')
                            else:
                                state.dfRawRangeShifted[tag] = state.dfRawRangeShifted[tag].shift(state.maximumidxKendall[indexTag], freq='min')
                        
                        elif fltMethodLag == "Mic":
                            state.dfRawRangeShifted[tag] = state.dfRawRangeShifted[tag].shift(state.maximumidxMic[indexTag], freq='min')
                           
                    #Shifta o dataset com base no maior do de best lag encontrando (que é o índice de maior valor das variáveis selecionadas para serem varridas pelo algoritmo de busca do best lag)
                    #newcolumns = newcolumns.columns.values[:-1]
                    #Função que fazia a mudança no nome da coluna, inserindo o sufixo _shifted, mas por causa do cache do streamlit, ocorrem bugs do streamlit e por isso, vamos fazer sem mudar o nome da coluna.
                    #for iterator in range(len(newcolumns)):
                    #    newcolumns[iterator] = newcolumns[iterator] + "_shifted"
                    #dfShiftedByBestlag = state.dfRawRange.copy()
                    #state.dfRawRangeShifted = dfShiftedByBestlag.shift(max(state.maximumidxPearson), freq='min') 
                    st.dataframe(state.dfRawRangeShifted)   
                    
                    state.dfRawRangeShifted.dropna(inplace=True)
                    
                    st.write("**Pronto!** Curtiu? Caso o Pandorabox tenha te ajudado de alguma forma, escreve pra gente e conta como foi ou se tá precisando de mais alguma coisa!")
                    st.write("Nosso e-mail: **inteligenciaindustrial@ihm.com.br**")

                    #st.write("**Pronto!** Agora estamos pronto para fazer uma modelagem mais robusta e eficiente. Vamos então para a aba **Data Modeling**")
                    
                    
                    '''
                    ##########################################################
                    ### TESTE - Numero de lag diferente para cada variavel ###
                    ### baseado no método escolhido pelo usuário           ###
                    ##########################################################
                    
                    st.title("Teste")
                    
                    testTags = fltTags.copy()
                    
                    fltSyncPearson = st.multiselect(label='Pearson', options=testTags, key='fltSyncPearson')
                    for a in fltSyncPearson:
                        if a in testTags:
                            testTags.remove(a)
                    
                    fltSyncSpearman = st.multiselect(label='Spearman', options=testTags, key='fltSyncSpearman')
                    for a in fltSyncSpearman:
                        if a in testTags:
                            testTags.remove(a)
                            
                    fltSyncKendall = st.multiselect(label='Kendall', options=testTags, key='fltSyncKendall')
                    for a in fltSyncKendall:
                        if a in testTags:
                            testTags.remove(a)
                            
                    fltSyncMIC = st.multiselect(label='MIC', options=testTags, key='fltSyncMIC')
                    for a in fltSyncMIC:
                        if a in testTags:
                            testTags.remove(a)
                            
                    teste = state.dfRawRange.copy()
                    
                    for tag in fltTags:
                            
                        indexTag = fltTags.index(tag)
                        
                        if tag in fltSyncPearson:
                            teste[tag] = teste[tag].shift(state.maximumidxPearson[indexTag], freq='min')
            
                        elif tag in fltSyncSpearman:
                            teste[tag] = teste[tag].shift(state.maximumidxSpearman[indexTag], freq='min')
                        
                        elif tag in fltSyncKendall:
                            teste[tag] = teste[tag].shift(state.maximumidxKendall[indexTag], freq='min')
                        
                        elif tag in fltSyncMIC:
                            teste[tag] = teste[tag].shift(state.maximumidxMic[indexTag], freq='min') 
                    
                    st.write(" ")
                    st.write(state.dfRawRange)

                    st.write(" ")
                    st.write(teste)           
                    
                    #methodLag = ["Pearson", "Spearman", "Kendall", "Mic"]
                    #a = {}
                    #for i in fltTags:
                    #    a[i] = st.selectbox(label=i, options=methodLag, key='fltSync{}'.format(i))
                    #st.write(a)
                    '''