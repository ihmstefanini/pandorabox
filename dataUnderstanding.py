# native imports
from google.protobuf.symbol_database import Default

# common imports
import numpy as np
import pandas as pd
from scipy.stats import linregress
import statsmodels.api as sm

# special imports
#from minepy import MINE
import ppscore 

# graphics
import matplotlib.pyplot as plt
import seaborn as sns

# front-end imports
import streamlit as st

# ===========================================
# Functions Data Retrieval
# ===========================================

### Realiza a transformação do arquivo csv em dataframe
@st.cache(allow_output_mutation=True)
def getDataFromCSV(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=",", decimal=".", encoding="UTF-8", index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    df = df.sort_index(ascending=True)
    df = df.apply(pd.to_numeric)
    return df

#@st.cache(allow_output_mutation=True)
def calcMic(df, mine):
    
    colunas = df.columns.values.tolist()

    zeros = np.zeros((len(colunas), len(colunas)))
    dfMic = pd.DataFrame(data=zeros, index=colunas, columns=colunas)

    for tagRemoved in colunas[:]:
        colunas.remove(tagRemoved)
        for tag in colunas:
            mine.compute_score(df[tag], df[tagRemoved]) 
            calcMic = mine.mic()
            dfMic.at[tag, tagRemoved] = calcMic 
    
    return dfMic

#@st.cache(allow_output_mutation=True)
def heatmapMIC(df, mine):  

    corrMic = calcMic(df, mine)
    maskCorr = np.triu(np.ones_like(corrMic, dtype=np.bool))                   

    fig, heatmap = plt.subplots()                     
    heatmap = sns.heatmap(corrMic, vmin=0, vmax=1, mask=maskCorr, annot=True, cmap="Reds")
    heatmap.set_title("Correlação de MIC", fontdict={'fontsize':18}, pad=12)
    
    return fig

@st.cache(allow_output_mutation=True)
def heatmapPPS(df):   

    matrix_df = ppscore.matrix(df, random_seed=123)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    
    fig, heatmap = plt.subplots() 
    heatmap = sns.heatmap(np.round(matrix_df, 2), vmin=0, vmax=1, cmap="Blues", annot=True)
    heatmap.set_title("Estimativa do 'Potencial preditivo' do problema", fontdict={'fontsize':12}, pad=12)
    
    return fig
    
@st.cache(allow_output_mutation=True)
def plotComparacaoCorrelacao(df, tag, target):  
    
    mine = MINE(alpha=0.6, c=15, est="mic_e")                     

    figScatter, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))   
    figScatter.suptitle("Comparação entre os métodos de correlação".format(tag, target), fontsize="large")
    axs[0].scatter(df[tag], df[target], s=10)
    axs[0].set_xlabel(tag)
    axs[0].set_ylabel(target)
    
    tempDf = df[[tag, target]]                   

    pearson = tempDf.corr(method='pearson').iloc[0][1]
    spearman =  tempDf.corr(method='spearman').iloc[0][1]
    kendall = tempDf.corr(method='kendall').iloc[0][1]
    
    mine.compute_score(df[tag], df[target]) 
    mic = mine.mic()
                        
    calcScatterCorr = [pearson, spearman, kendall, mic]
    methodScatterCorr = ["pearson", "spearman", "kendall", "mic"]

    sns.barplot(x=methodScatterCorr, y=calcScatterCorr, palette="deep", ax=axs[1])
    axs[1].set_ylim(-1,1)
    axs[1].grid(which='major', axis="y")
    
    return figScatter        
    
def write(state):
    
    st.title('Avaliando as correlações e o potencial preditivo')
    st.markdown('Após prepararmos os dados, vamos agora analisar os **cálculos de correlação** (etapa anterior a de sincronização).')        

    if type(state.dfRawRange) == type(None):
        
        st.warning("Realize a importação do arquivo CSV contendo os dados em **'Data Preparation'**.")
                
    elif state.dfRawRange.empty:
        
        st.text("Dados carregados com sucesso!")
        st.warning("Realize os passos 1 e 2 do **'Data Preparation'**.")
    
    else:
        
        #st.title("Data Correlation")
        st.markdown('Verifique os **passos 1 e 2** na aba lateral à esquerda ')
        #st.text("Dados carregados com sucesso!")
        st.sidebar.title("Data Correlation")
        
        ### Seleção das variáveis para análise
        expanderVarCorr = st.sidebar.beta_expander("Passo 1: Selecione a variável", expanded=False)
                    
        state.fltTarget = expanderVarCorr.selectbox(
                label='Variável alvo do cálculo de correlação:', 
                options=state.fltTags,
                index=state.fltTags.index(state.fltTarget) if state.fltTarget else 0,
                key='fltTargetUnderstanding')
        
        if len(state.fltTags) >= 2:
                
            state.fltPredictor = state.fltTags.copy()
            state.fltPredictor.remove(state.fltTarget)
                
        ### Correlação simples
        expanderMethodCorr = st.sidebar.beta_expander("Passo 2: Selecione o método para a correlação", expanded=False)
        optionCorr=["pearson", "kendall", "spearman"]
        state.methodCorr = expanderMethodCorr.selectbox(label='Método de cáclulo da Correlação', options=optionCorr, index=optionCorr.index(state.methodCorr) if state.methodCorr else 0)                       
        
        if len(state.fltTags) > 0:
            st.markdown('Selecione as opções abaixo **(uma por vez)** para avaliar os dados:')

            showRawTimeSeries = st.checkbox(label='Visualizar as séries temporais das variáveis', value=False, key='showRawTimeSeries')
            if (showRawTimeSeries):
                fltSelectedVar = st.multiselect(label='Escolha a variável de processo que deseja visualizar:', options=state.fltTags, key='fltSelectedVar')
                if (len(fltSelectedVar) > 0):                       
                    st.line_chart(data=state.dfRawRange[fltSelectedVar], width=1000, height=150, use_container_width=True)
                    st.line_chart(data=state.dfRawRange[state.fltTarget], width=1000, height=150, use_container_width=True)
            
            if len(state.fltTags) < 2:
                 st.warning("Selecione no mínimo duas variaveis no passo 1 do **'Data Preparation'** para análise.")
               
            # Realiza a correlação do dataframe filtrado pelas variaveis selecionadas e intervalo de tempo
            showRawCorr = st.checkbox(label='Visualizar as correlações lineares', value=False, key='showRawCorr')
            if (showRawCorr) & (len(state.fltTags) >= 2):
                
                #########################
                ### Correlação Linear ###
                #########################
                
                #st.subheader("Métodos Lineares")   
                maskCorr = np.triu(np.ones_like(state.dfRawRange.corr(method=state.methodCorr), dtype=np.bool))
                
                figCorr, heatmapCorr = plt.subplots() 
                heatmapCorr = sns.heatmap(np.round(state.dfRawRange.corr(method=state.methodCorr), 2), vmin=-1, vmax=1, mask=maskCorr, annot=True, cmap="coolwarm")
                heatmapCorr.set_title("Correlação de {}".format(state.methodCorr), fontdict={'fontsize':12}, pad=12)
                st.pyplot(figCorr)

                st.text(" ")
                st.text("Um breve guia sobre análise de correlação:")
                st.info(
                    """
                    Análise de correlação remete ao quanto que uma variável está correlacionada com outra linearmente e o sentido da linearidade.
                    \nValores variam entre -1 e 1 (valores mínimo e máximo), sendo que 1 significa que quando uma variável aumenta a outra também e -1 o inverso.
                    \nQuanto mais correlacionada as variáveis estão (seja mais próximo de -1 ou de 1), maiores são as chances de se obter um modelo de machine learning que capture a relação linear entre ambas!
                    """
                )
                st.subheader('')

                st.text(" ")
                st.text("Cuidados com a análise de correlação:")
                st.info(
                    """
                    Apesar de ser muito útil, não é recomendado utilizar somente os métodos de correlação para julgar o potencial preditivo de um problema.
                    \nÉ recomendável avaliar outros critérios para se ter uma visão mais holística do potencial preditivo.
                    \n**Selecione** as demais opções para obter mais informação do potencial preditivo do problema que está explorando
                    """
                )
                st.subheader('')

                #################################
                ### Análise Linear vs Pearson ###
                #################################
                
                #if False:
                if (state.methodCorr == "pearson"):
                    
                    showLinearCorr = st.checkbox(label='Visualizar análises complementares à de correlação linear de pearson', value=False, key='showLinearCorr')
                    if (showLinearCorr == True):
                        #st.title("Análises complementares de Rˆ2")
                        
                                           
                        varAnalise = st.selectbox(label='Primeiro, selecione a variável para realizarmos as análises complementares:', options=state.fltTags, key='varAnalise')                    

                        ########################
                        ### Regressão Linear ###
                        ########################
                        
                        if not state.dfRawRange[state.fltTarget].isnull().values.any():
                            regLinear = linregress(state.dfRawRange[varAnalise], state.dfRawRange[state.fltTarget])
                            resultRegLinear = regLinear.intercept + regLinear.slope*state.dfRawRange[varAnalise]
                            
                            figRegLinear = plt.figure()
                            plt.plot(state.dfRawRange[varAnalise], state.dfRawRange[state.fltTarget], 'o', alpha=0.1, markersize=4)
                            plt.plot(state.dfRawRange[varAnalise], resultRegLinear, alpha=0.6)
                            plt.xlabel('{}'.format(varAnalise))
                            plt.ylabel('{}'.format(state.fltTarget))
                            plt.title('Scatter plot of {} versus {}'.format(varAnalise, state.fltTarget))
                            st.pyplot(figRegLinear)
                            
                            ################
                            ### Resíduos ###
                            ################
                            
                            #st.markdown("**Insights:**")
                            
                            resid = state.dfRawRange[state.fltTarget] - resultRegLinear
                            a = np.corrcoef(state.dfRawRange[varAnalise], state.dfRawRange[state.fltTarget])
                            rho_actual = a[0, 1]
                            R2 = 1 - resid.var()/state.dfRawRange[state.fltTarget].var()
                            
                            st.subheader("Análise complementar 1 - Resíduos (por meio de regressão linear)") 
                            st.info("Resíduos são os erros de cada previsão e são utilizados para calcular o famoso **coeficiente de determinação**, mais conhecido como **$R^2$**.\
                                    \n\n$R^2$ = {:.2f}, conforme a seleção das variáveis (experimente outras variáveis para observar outros resultados)".format(R2))

                            st.info("O $R^2$ mede a parte da variância na variável dependente **{}** que é *'explicada'* pelo preditor **{}**.\
                                    \n\nOu seja, se usarmos **{}** para predizer **{}**, a variância dos erros será **{:.2f}%** menor do que a variância de **{}**.\
                                    \n\nIsso soa menos impressionante do que uma correlação de **{:.2f}**, pois existe uma relação entre a correlação e o coeficiente de determinação:\
                                    \n\n$R^2 = \\rho^2$\
                                    \n\nOu seja, o coeficiente de determinação é a correlação ao quadrado."\
                                    .format(state.fltTarget, varAnalise, varAnalise, state.fltTarget,  R2*100, state.fltTarget, rho_actual)
                                    )
                            '''
                            st.info("Sendo a correlação menor que 1, $R^2$ geralemtne é menor que $\\rho$.\
                                    \n\n**{}**: {:.2f}\
                                    \n\n**{}**: {:.2f}".format("R2", R2,"$\\rho$ actual", rho_actual))
                            '''
                            st.info("Se você tiver a opção de relatar $R^2$ ou correlação, é sugerido que informe $R^2$ porque é mais significativo (redução percentual na variância) e menos **falsamente impressionante!**.\
                                    \n\nNo entanto, **$R^2$ também é problemático**, porque reduzir a variância geralmente não é o que nós preocupa.\
                                    \n\nSe o objetivo é **quantificar a qualidade de uma previsão**, é melhor usar uma **métrica de erro** que signifique algo no contexto do problema.")
                            
                            ###########################
                            ### Erro médio absoluto ###
                            ###########################
                            
                            st.subheader("Análise complementar 2 - Erro Médio Absoluto")
                            
                            MAE_after = np.abs(resid).std()
                            deviation = state.dfRawRange[state.fltTarget] - state.dfRawRange[state.fltTarget].mean()
                            
                            MAE_before = np.abs(deviation).std()
                            improvementMAE = 1 - MAE_after / MAE_before
                            improvementMAE_percent = (1 - MAE_after / MAE_before)*100
                            
                            st.info("Uma segunda opção é avaliar também o **erro médio absoluto** (ou MAE), que é exatamente o que diz: a média dos valores absolutos dos resíduos.\
                                    \n\n**MAE dos resíduos após a predição por regressão linear:** {:.2f}".format(MAE_after))

                            st.info("Se você utilizar a variável **{}** para predizer a variável alvo **{}**, deverá ter um erro em unidades de engenharia de aproximadamente {:.2f} em média. \
                                    \n\nUma maneira de colocar isso em contexto é compará-lo ao **MAE** se não soubermos a variável **{}**.\
                                    \n\nNesse caso, a melhor estratégia é 'adivinhar' a média da variável alvo todas as vezes. \
                                    \n\n**MAE da própria variável alvo (sem fazer regressão linear):** {:.2f}"\
                                    .format(varAnalise, state.fltTarget, MAE_after, varAnalise, MAE_before))
                            #corrigir ou retirar essa linha, dando um exemplo pro caso corrente... usando dados das variáveis em estudo    
                            st.info("Se você sempre acertar 100, deve esperar um erro de cerca de 8,5 pontos em média. \
                                    \n\nPodemos usar esses resultados para calcular a melhoria percentual no MAE, com e sem pontuação SAT: \
                                    \n\n**Melhoria do MAE:** {:.2f}\
                                    \n\nPortanto, podemos dizer que utilizar a variável **{} diminui o MAE em {:.1f}%**.\
                                    \n\nIsso é uma melhoria, porém menor do que R2 = {:.2f} e menor do que ρ = {:.2f}.".format(improvementMAE, varAnalise, improvementMAE_percent, R2, rho_actual))

                            #st.info("Portanto, podemos dizer que utilizar a variável **{} diminui o MAE em {:.1f}%**.\
                            #        \n\nIsso é uma melhoria, porém menor do que R2 = {:.2f} e menor do que ρ = {:.2f}."\
                            #        .format(varAnalise, improvementMAE_percent, R2, rho_actual))

                            ############
                            ### RMSE ###
                            ############

                            st.subheader("Análise complementar 3 - RMSE")
                            
                            RMSE_after = resid.std()
                            RMSE_before = state.dfRawRange[state.fltTarget].std() #brenão, estava varAnalise, mas aqui é o target!
                            improvementRMSE = 1 - RMSE_after / RMSE_before
                            improvementRMSE_percent = (1 - RMSE_after / RMSE_before)*100
                            st.info("Uma terceira opção é usar a raiz quadrada do erro quadrático (RMSE - root mean squared error), que nada mais é que o desvio padrão dos resíduos:\
                                    \n\n**RMSE dos resíduos após a predição por regressão linear:** {:.2f}".format(RMSE_after))

                            st.info("Pode-se comparar esse resultado em relação ao valor do RMSE sem utilizar a regressão linear (usando somente a própria variável alvo para prever ela mesma), que é o desvio padrão da variável alvo **{}**:\
                                    \n\n**RMSE da própria variável alvo (sem fazer regressão):** {:.2f}".format(state.fltTarget, RMSE_before))

                            st.info("E agora podemos calcular a melhoria usando o RMSE:\
                                    \n\n**Melhoria usando RMSE** {:.3f}\
                                    \n\nEntão, se usarmos a variável **{}**, podemos **reduzir o RMSE em {:.1f}%**.".format(improvementRMSE, varAnalise, improvementRMSE_percent))

                            
                            #Texto reduzido nesse último st.info para facilitar a intepretação do user
                            #\n\nThere is no compelling reason to prefer RMSE over MAE, but it has practical one advantage: we don't need the data to compute the RMSE.  We can derive it from the variance of IQ and $R^2$:\
                            #        \n\n$R^2 = 1 - Var(resid) ~/~ Var(iq)$\
                            #        \n\n$Var(resid) = (1 - R^2)~Var(iq)$\
                            #        \n\n$Std(resid)$ = $\sqrt((1 - R^2)~Var(iq))$\
                            #        \n\n**STD(resid)**: {:.2f}\
                            #        \n\n**RMSE_after**: {:.2f}"
                            #        .format( np.sqrt((1-R2) * state.dfRawRange[state.fltTarget].var()), RMSE_after))

                            ########################
                            ### Percentage error ###
                            ########################
                            
                        #     st.subheader("Análise complementar 4 - Erro percentual")
                            
                        #     deviation = state.dfRawRange[state.fltTarget] - state.dfRawRange[state.fltTarget].mean() # Brenao tinha colocado a varAnalise, corrigido para fltTarget
                        #     MAPE_before = np.abs(deviation / state.dfRawRange[state.fltTarget]).mean() * 100
                        #     MAPE_after = np.abs(resid / state.dfRawRange[state.fltTarget]).mean() * 100
                        #     improvementMAPE = 1 - MAPE_after / MAPE_before
                            
                        #     st.info("A quarta forma de expressar o potencial de se usar a variável **{}** para predizer o target {} é avaliar o erro absoluto percentual (MAPE).\
                        #             \n\nNovamente, se não temos a variável {}, a melhor estratégia é adivinhar a média. Nesse caso, o MAPE é:\
                        #             \n\n**MAPE da própria variável alvo (sem regressão)**: {:.2f}".format(varAnalise, state.fltTarget, varAnalise, MAPE_before))
                            
                        #     st.info("Se sempre adivinharmos a média, espera-se um erro de aproximadamente {:.2f}, na média.\
                        #             \n\nSe usarmos a variável {} para fazermos melhores estimativas, o  MAPE é menor:\
                        #             \n\n**MAPE dos resíduos**: {:.2f}".format(MAPE_before, varAnalise, MAPE_after)) 
                            
                        #     st.info("So we expect to be off by {:.2f}% on average.\
                        #             \n\nAnd we can quantify the improvement like this:\
                        #             \n\n**improvementMAPE**: {:.2f}".format(MAPE_after-MAPE_before,improvementMAPE))
                            
                        #     st.info("Usando **{}** para predizer {} reduz o erro médio absoluto percentual em {:.1f}.\
                        #             \n\nI included MAPE in this discussion because it is a good choice in some contexts, but this is probably not one of them.\
                        #             \n\nUsing MAPE implies that an error of 1 IQ point is more important for someone with low IQ and less important for someone with high IQ. In this context, it's not clear whether that's true.")
                            
                            ###############            
                            ### Summary ###           
                            ###############                                
                                    
                            st.subheader("Resumo das análises")                     
                            
                            st.info(
                                "Em suma, correlação é uma problemática estatística pois pode soar mais impressionante do que realmente é.\
                                \n\nCoeficiente de determinação, $R^2$, seria uma melhor alternativa por trazer uma interpretação mais natural: percentual de redução na variância. Porém, redução de variância não o que usualmente queremos com uma predição.\
                                \n\nPortanto, avalie se no seu caso não seria melhor escolher uma medida do erro que tem algum significado no contexto do seu problema, possivelmente uma das métricas:\
                                \n* MAE: Mean absolute error\
                                \n* RMSE: Root mean squared error\
                                \n\nQual desses é mais signigicativo, vai depender da função de custo a ser otimizada para o seu problema.  O custo de estar errado depende do erro absoluto ou do erro quadrático?  Se sim, isso deveria guiar a sua escolha.\
                                \n\nNesse exemplo, a correlação é **{:.2f}**, que parece ser mais impressionante do que é.\
                                \n\n$R^2$ é **{:.2f}**, o que significa que podemos reduzir a variância por **{:.2f}**%. Mas também parece mais impressionante do que realmente é.\
                                \n\nUtilizando {} para predizer {}, podemos reduzir:\
                                \n* $R^2$ em {:.2f}%\
                                \n* MAE em {:.2f}%\
                                \n* RMSE em {:.2f}%\
                                \n\nReportar qualquer uma dessas métricas de erro é mais significativo que reportar uma correlação ou um $R^2$ de tanto."\
                                .format(rho_actual,R2, R2*100, varAnalise, state.fltTarget, R2*100, improvementMAE*100,improvementRMSE*100)
                            )
                        
                        else:
                            st.warning("A variável em análise possui valores **NaN**, realize a limpeza em **Data Preparation**.")

                        ###########
                         ### PPS ###
                        ###########
                        
                        #st.subheader("Poder Preditivo - PPS")
                        #figPPS = heatmapPPS(state.dfRawRange)
                        #st.pyplot(figPPS, fontdict={'fontsize':12}, pad=12) 


             
                
                ###########
                ### MIC ###
                ###########
                
                #mine = MINE(alpha=0.6, c=15, est="mic_approx")
                
                #st.subheader("Correlação Não-Linear")
                #figMicCorr = heatmapMIC(state.dfRawRange, mine)
                #st.pyplot(figMicCorr)
                
                ###########
                ### PPS ###
                ###########
                
                #st.subheader("Poder Preditivo - PPS")
                #figPPS = heatmapPPS(state.dfRawRange)
                #st.pyplot(figPPS)
            

            showStatisticalSignificance = st.checkbox(label='Visualizar Teste Estatístico de Significância', value=False, key='showStatisticalSignificance')    
            if (showStatisticalSignificance):
                fltSelectedVarStatisPreditora = st.multiselect(label='Escolha a(s) variável(s) independente(s):', options=state.fltTags, key='fltSelectedVarStatisPreditora')
                if (len(fltSelectedVarStatisPreditora) > 0):                       
                    list_independent_variables = []
                    for i in range(0,len(fltSelectedVarStatisPreditora)):
                        list_independent_variables.append(fltSelectedVarStatisPreditora[i])
                    X2 = sm.add_constant(state.dfRawRange[list_independent_variables])
                    est = sm.OLS(state.dfRawRange[state.fltTarget], X2)
                    est2 = est.fit()
                    p_valor_variavel = []
                    for i in range(0,len(fltSelectedVarStatisPreditora)):
                        p_valor_variavel.append(est2.pvalues[i+1])
                    # Se rejeitar hipotese nula, coeficiente é significativo
                    for i in range(0,len(fltSelectedVarStatisPreditora)):
                        if(p_valor_variavel[i]<0.05):
                            st.markdown("A variável " + str(fltSelectedVarStatisPreditora[i]) + " é estatisticamente significante para a previsão de " + str(state.fltTarget))
                            st.markdown("Your data favor the hypothesis that there is a non-zero correlation. Changes in the independent variable are associated with changes in the response at the population level. This variable is statistically significant and probably a worthwhile addition to your regression model")
                        else:
                            st.markdown("A variável " + str(fltSelectedVarStatisPreditora[i]) + " não é estatisticamente significante para a previsão de " + str(state.fltTarget))
                            st.markdown("There is insufficient evidence in your sample to conclude that a non-zero correlation exists")
                        
            if (showRawCorr) & (len(state.fltTags) < 2):
                st.warning("Selecione no mínimo duas variaveis no passo 1 do **'Data Preparation'** para análise.")

            #if (showRawCorr) or (showRawTimeSeries):
            #    if len(fltSelectedVar) > 0:
            #        st.markdown('4. Muito bem! Chegamos a fim da primeira parte da nossa jornada. Agora selecione a próxima aba, a **Data Syncronization**, para continuarmos nossa jornada!')    


        #     showScatterplotCorr = st.checkbox(label='Scatter plot', value=False, key='showScatterplotCorr')
        #     if (showScatterplotCorr) & (len(state.fltTags) >= 2):
                                
        #         mine = MINE(alpha=0.6, c=15, est="mic_approx") 
                
        #         fltTags = state.dfRawRange.columns.values.tolist()
        #         fltTags.remove(state.fltTarget)
        #         #figScatter, axs = plt.subplots(nrows=len(state.fltTags), ncols=2)                
                
        #         for i in range(len(fltTags)):
        #             st.write("")
        #             figScatter, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))   
        #             figScatter.suptitle("Comparação entre os métodos de correlação".format(fltTags[i], state.fltTarget), fontsize="large")
        #             sns.scatterplot(x=state.dfRawRange[fltTags[i]], y=state.dfRawRange[state.fltTarget], data=state.dfRawRange[fltTags[i]], ax=axs[0])

        #             pearson = pearson(state.dfRawRange[fltTags[i]], state.dfRawRange[state.fltTarget])[0]
        #             spearman =  spearman(state.dfRawRange[fltTags[i]], state.dfRawRange[state.fltTarget]).correlation
        #             kendall = kendalltau(state.dfRawRange[fltTags[i]], state.dfRawRange[state.fltTarget]).correlation
                    
        #             mine.compute_score(state.dfRawRange[fltTags[i]], state.dfRawRange[state.fltTarget]) 
        #             mic = mine.mic()
                    
        #             pps = ppscore.score(state.dfRawRange, fltTags[i], state.fltTarget, sample=5_000, cross_validation=4, random_seed=123, invalid_score=0, catch_errors=True)['ppscore']
                    
        #             #calcScatterCorr = [pearson, spearman, kendall]
        #             #methodScatterCorr = ["pearson", "spearman", "kendall"]
                    
        #             calcScatterCorr = [pearson, spearman, kendall, mic, pps]
        #             methodScatterCorr = ["pearson", "spearman", "kendall", "mic", "pps"]

        #             sns.barplot(x=methodScatterCorr, y=calcScatterCorr, palette="deep", ax=axs[1])
        #             axs[1].set_ylim(-1,1)
        #             axs[1].grid(which='major', axis="y")
            
        #             st.pyplot(figScatter)


            