# native imports
from google.protobuf.symbol_database import Default

# common imports
import numpy as np
import pandas as pd
from scipy.stats import linregress
import statsmodels.api as sm

# special imports
from minepy import MINE
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

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def calcMic(df):
    
    mine = MINE(alpha=0.6, c=15, est="mic_e")
    
    colunas = df.columns.values.tolist()

    zeros = np.zeros((len(colunas), len(colunas)))
    dfMic = pd.DataFrame(data=zeros, index=colunas, columns=colunas)

    for tagRemoved in colunas[:]:
        
        colunas.remove(tagRemoved)       
        
        for tag in colunas:
            
            mine.compute_score(df[tag], df[tagRemoved]) 
            dfMic.at[tag, tagRemoved] = mine.mic()

    return dfMic

def heatmapMIC(df):  

    corrMic = calcMic(df)
    maskCorr = np.triu(np.ones_like(corrMic, dtype=np.bool))                   

    fig, heatmap = plt.subplots()                     
    heatmap = sns.heatmap(np.round(corrMic,2), vmin=0, vmax=1, mask=maskCorr, annot=True, cmap="Reds")
    heatmap.set_title("Correlação pelo método MIC", fontdict={'fontsize':18}, pad=12)
    
    return fig

@st.cache(allow_output_mutation=True)
def heatmapPPS(df):   

    matrix_df = ppscore.matrix(df, random_seed=123)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    
    fig, heatmap = plt.subplots() 
    heatmap = sns.heatmap(np.round(matrix_df, 2), vmin=0, vmax=1, cmap="Blues", annot=True)
    heatmap.set_title("Estimativa do 'Potencial preditivo' do problema", fontdict={'fontsize':18}, pad=12)
    
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
    
    st.title('Obtendo os dados correlacionados')
    st.markdown('Após prepararmos os dados, vamos agora analisar os **cálculos de correlação** (etapa anterior a de sincronização).')        

    if type(state.dfRawRange) == type(None):
        
        st.warning("Realize a importação do arquivo CSV contendo os dados em **'Data Preparation'**.")
                
    elif state.dfRawRange.empty:
        
        st.text("Dados carregados com sucesso!")
        st.warning("Realize os passos 1 e 2 do **'Data Preparation'**.")
    
    else:
        
        #st.title("Data Correlation")
        st.markdown('Agora siga os **passos na aba lateral à esquerda** ')
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
            st.markdown('3. Pronto! Agora selecione as opções abaixo **(uma por vez)** para visualizar:')

            showRawTimeSeries = st.checkbox(label='Séries temporais das variáveis', value=False, key='showRawTimeSeries')
            if (showRawTimeSeries):
                fltSelectedVar = st.multiselect(label='Escolha a variável de processo que deseja visualizar:', options=state.fltTags, key='fltSelectedVar')
                if (len(fltSelectedVar) > 0):                       
                    st.line_chart(data=state.dfRawRange[fltSelectedVar], width=1000, height=150, use_container_width=True)
                    st.line_chart(data=state.dfRawRange[state.fltTarget], width=1000, height=150, use_container_width=True)
            
            if len(state.fltTags) < 2:
                 st.warning("Selecione no mínimo duas variaveis no passo 1 do **'Data Preparation'** para análise.")
               
            # Realiza a correlação do dataframe filtrado pelas variaveis selecionadas e intervalo de tempo
            showRawCorr = st.checkbox(label='Visualizar correlações (métodos lineares)', value=False, key='showRawCorr')
            if (showRawCorr) & (len(state.fltTags) >= 2):
                
                #########################
                ### Correlação Linear ###
                #########################
                
                #st.subheader("Métodos Lineares")   
                maskCorr = np.triu(np.ones_like(state.dfRawRange.corr(method=state.methodCorr), dtype=np.bool))
                
                figCorr, heatmapCorr = plt.subplots() 
                heatmapCorr = sns.heatmap(np.round(state.dfRawRange.corr(method=state.methodCorr), 2), vmin=-1, vmax=1, mask=maskCorr, annot=True, cmap="coolwarm")
                heatmapCorr.set_title("Correlação de {}".format(state.methodCorr), fontdict={'fontsize':18}, pad=12)
                st.pyplot(figCorr)
                
                st.info(
                    """
                    Análise de correlação remete ao quanto que uma variável está correlacionada com outra linearmente e o sentido da linearidade.
                    \nValores variam entre -1 e 1 (valores máximos), sendo que 1 significa que quando uma variável aumenta a outra também e -1 o inverso.
                    \nQuanto mais correlacionada as variáveis estão (seja mais próximo de -1 ou de 1), maiores são as chances de se obter um modelo de machine learning que capture a relação linear entre ambas!
                    """
                )
                st.subheader('')

                #################################
                ### Análise Linear vs Pearson ###
                #################################
                
                #if False:
                if (state.methodCorr == "pearson"):
                    
                    showLinearCorr = st.checkbox(label='Visualizar análises complementares à de Pearson', value=False, key='showLinearCorr')
                    if (showLinearCorr == True):
                        st.title("Análises complementares de Rˆ2")
                        
                        st.subheader("Regressão Linear")                    
                        varAnalise = st.selectbox(label='Variável para a análise linear:', options=state.fltTags, key='varAnalise')                    

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
                            
                            st.title("Residuals")
                            
                            resid = state.dfRawRange[state.fltTarget] - resultRegLinear
                            a = np.corrcoef(state.dfRawRange[varAnalise], state.dfRawRange[state.fltTarget])
                            rho_actual = a[0, 1]
                            R2 = 1 - resid.var()/state.dfRawRange[state.fltTarget].var()
                            
                            st.info("Resíduos são os erros de cada previsão e são utilizados para calcular o coeficiente de determinação, $R^2$.\
                                    \n\n$R^2$ = {:.2f}".format(R2))

                            st.info("$R^2$ mede a parte da variância na variável dependente **{}** que é *'explicada'* pelo preditor **{}**.\
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
                            st.info("Se você tiver a opção de relatar $R^2$ ou correlação, é sugerido que informe $R^2$ porque é mais significativo (redução percentual na variância) e menos falsamente impressionante.\
                                    \n\nNo entanto, $R^2$ também é problemático, porque reduzir a variância geralmente não é o que nós preocupa.\
                                    \n\nSe o objetivo é quantificar a qualidade de uma previsão, é melhor usar uma métrica de erro que signifique algo no contexto do problema.")
                            
                            ###########################
                            ### Erro médio absoluto ###
                            ###########################
                            
                            st.title("Erro Médio Absoluto")
                            
                            MAE_after = np.abs(resid).std()
                            deviation = state.dfRawRange[state.fltTarget] - state.dfRawRange[state.fltTarget].mean()
                            
                            MAE_before = np.abs(deviation).std()
                            improvementMAE = 1 - MAE_after / MAE_before
                            
                            st.info("Uma opção é o **erro médio absoluto**, que é exatamente o que diz: a média dos valores absolutos dos resíduos.\
                                    \n\n**MAE_after:** {:.2f}".format(MAE_after))

                            st.info("Se você utilizar o **{}** para predizer o **{}**, deverá ter uma perda de cerca de *5* pontos em média. \
                                    \n\nUma maneira de colocar isso em contexto é compará-lo ao **MAE** se não soubermos **{}**.\
                                    \n\nNesse caso, a melhor estratégia é adivinhar a média todas as vezes. \
                                    \n\n**MAE_before:** {:.2f}"\
                                    .format(varAnalise, state.fltTarget, varAnalise, MAE_before))
                        
                            st.info("Se você sempre acertar 100, deve esperar um erro de cerca de 8,5 pontos em média. \
                                    \n\nPodemos usar esses resultados para calcular a melhoria percentual no MAE, com e sem pontuação SAT: \
                                    \n\n**Melhoria MAE:** {:.2f}".format(improvementMAE))

                            st.info("Portanto, podemos dizer que conhecer as pontuações do SAT diminui o MAE em 44%.\
                                    \n\nIsso certamente é uma melhoria, mas observe que parece menos impressionante do que R2 = 0,66 e muito menos impressionante do que ρ = 0,82.")

                            ############
                            ### RMSE ###
                            ############

                            st.title("RMSE")
                            
                            RMSE_after = resid.std()
                            RMSE_before = state.dfRawRange[varAnalise].std()
                            improvementRMSE = 1 - RMSE_after / RMSE_before
                            
                            st.info("Another option is RMSE (root mean squared error) which is the standard deviation of the residuals:\
                                    \n\n**RMSE_after:** {:.2f}".format(RMSE_after))

                            st.info("We can compare that to RMSE without SAT scores, which is the standard deviation of IQ:\
                                    \n\n**RMSE_before:** {:.2f}".format(RMSE_before))

                            st.info("And here's the improvement:\
                                    \n\n**Improvement RMSE** {:.2f}".format(improvementRMSE))

                            st.info("If you know someone's SAT score, you can decrease your RMSE by 42%.\
                                    \n\nThere is no compelling reason to prefer RMSE over MAE, but it has practical one advantage: we don't need the data to compute the RMSE.  We can derive it from the variance of IQ and $R^2$:\
                                    \n\n$R^2 = 1 - Var(resid) ~/~ Var(iq)$\
                                    \n\n$Var(resid) = (1 - R^2)~Var(iq)$\
                                    \n\n$Std(resid)$ = $\sqrt((1 - R^2)~Var(iq))$\
                                    \n\n**STD(resid)**: {:.2f}\
                                    \n\n**RMSE_after**: {:.2f}".format(np.sqrt((1-R2) * state.dfRawRange[state.fltTarget].var()), RMSE_after) )
                                    #"(1 - R^2) Var(iq)"

                            ########################
                            ### Percentage error ###
                            ########################
                            
                            st.title("Percentage error")
                            
                            deviation = state.dfRawRange[varAnalise] - state.dfRawRange[varAnalise].mean()
                            MAPE_before = np.abs(deviation / state.dfRawRange[varAnalise]).mean() * 100
                            MAPE_after = np.abs(resid / state.dfRawRange[state.fltTarget]).mean() * 100
                            improvementMAPE = 1 - MAPE_after / MAPE_before
                            
                            st.info("One other way to express the value of SAT scores for predicting IQ is the mean absolute percentage error (MAPE).\
                                    \n\nAgain, if we don't know SAT scores, the best strategy is to guess the mean. In that case the MAPE is:\
                                    \n\n**MAPE_before**: {}".format(MAPE_before))
                            
                            st.info("If we always guess the mean, we expect to be off by 12%, on average.\
                                    \n\nIf we use SAT scores to make better guesses, the MAPE is lower:\
                                    \n\n**MAPE_after**: {}".format(MAPE_after)) 
                            
                            st.info("So we expect to be off by 6.6% on average.\
                                    \n\nAnd we can quantify the improvement like this:\
                                    \n\n**improvementMAPE**: {}".format(improvementMAPE))
                            
                            st.info("Using SAT scores to predict IQ decreases the mean absolute percentage error by 42%.\
                                    \n\nI included MAPE in this discussion because it is a good choice in some contexts, but this is probably not one of them.\
                                    \n\nUsing MAPE implies that an error of 1 IQ point is more important for someone with low IQ and less important for someone with high IQ. In this context, it's not clear whether that's true.")
                            
                            ###############            
                            ### Summary ###           
                            ###############                                
                                    
                            st.title("Summary")                     
                            
                            st.info(
                                "Correlation is a problematic statistic because it sounds more impressive than it is.\
                                \n\nCoefficient of determination, $R^2$, is better because it has a more natural interpretation: percentage reduction in variance.  But reducing variance it usually not what we care about.\
                                \n\nI think it is better to choose a measurement of error that is meaningful in context, possibly one of:\
                                \n* MAE: Mean absolute error\
                                \n* RMSE: Root mean squared error\
                                \n* MAPE: Mean absolute percentage error\
                                \n\nWhich one of these is most meaningful depends on the cost function.  Does the cost of being wrong depend on the absolute error, squared error, or percentage error?  If so, that should guide your choice.\
                                \n\nOne advantage of RMSE is that we don't need the data to compute it; we only need the variance of the dependent variable and either $\\rho$ or $R^2$.\
                                \n\nIn this example, the correlation is **{:.2f}**, which sounds much more impressive than it is.\
                                \n\n$R^2$ is **{:.2f}**, which means we can reduce variance by **{:.2f}**%.  But that also sounds more impressive than it is.\
                                \n\nUsing {} to predict {}, we can reduce:\
                                \n* $R^2$ by {:.2f}%\
                                \n* MAE by {:.2f}%\
                                \n* RMSE by {:.2f}%\
                                \n* MAPE by {:.2f}%.\
                                \n\nReporting any of these is more meaningful than reporting correlation or $R^2$."\
                                .format(rho_actual,R2, R2*100, varAnalise, state.fltTarget, R2*100, improvementMAE*100,improvementRMSE*100,improvementMAPE*100 )
                            )
                        
                        else:
                            st.warning("A variável em análise possui valores **NaN**, realize a limpeza em **Data Preparation**.")
                            
                ###########
                ### MIC ###
                ###########

                st.subheader("Correlação Não-Linear")
                
                showCalcMIC = st.checkbox(label='Cálculo da correlação MIC', value=False, key='showCalcMIC')
                st.warning("O cálculo do MIC demanda de grande esforço computacional, portanto a execução do cálculo pode levar algum tempo para a finalização.")
                
                if showCalcMIC == True:
                    
                    figMicCorr = heatmapMIC(state.dfRawRange)
                    st.pyplot(figMicCorr)
                
                ###########
                ### PPS ###
                ###########
                
                st.subheader("Poder Preditivo - PPS")
                figPPS = heatmapPPS(state.dfRawRange)
                st.pyplot(figPPS)
