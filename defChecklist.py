# Defs Checklist
import numpy as np

# Avalia o modo da malha
def modoMalha(malha, df):
    
    modos = df["{}_MODO".format(malha)]    
    values, counts = np.unique(modos, return_counts=True)
    
    return "{} - {:.2f}%".format(values[np.argmax(counts)], max(counts)/len(modos)*100)

# Calcula o erro da malha através da normalização do Erro Médio Absoluto
def erroMalha(malha, df):
        
    dfDifMan = df[df["{}_MODO".format(malha)] != "{}_MANUAL".format(malha)]
    
    dfDifMan["Erro"] = dfDifMan["{}_SP".format(malha)] - dfDifMan["{}_PV".format(malha)]

    q1 = dfDifMan["Erro"].quantile(q=0.25)
    q3 = dfDifMan["Erro"].quantile(q=0.75)
    iqr = q3 - q1
    
    meta = q1 - 1.5*iqr
    limiar = q3 + 1.5*iqr
    divisor = limiar - meta

    erroSel = dfDifMan[(dfDifMan["Erro"] >= meta) & (dfDifMan["Erro"] <= limiar)]["Erro"] 

    erroMeanAbs = abs(erroSel).mean()
    
    erroNorm = (erroMeanAbs-meta)/divisor
    percErroNorm = np.round(erroNorm*100, 2)
    
    return "{:.2f}%".format(percErroNorm)

# Calcula a porcentagem de saturação para os limites passados
def saturacao(malha, df, cvMin, cvMax):
    
    modo = df[df["{}_MODO".format(malha)] != "{}_MANUAL".format(malha)].count()["{}_MODO".format(malha)]
    satSup = df[(df["{}_MODO".format(malha)] != "{}_MANUAL".format(malha)) & (df["{}_CV".format(malha)] >= cvMax)].count()["{}_CV".format(malha)]
    satInf = df[(df["{}_MODO".format(malha)] != "{}_MANUAL".format(malha)) & (df["{}_CV".format(malha)] <= cvMin)].count()["{}_CV".format(malha)]
    
    porcSatSup = (satSup/modo)*100
    porcSatInf = (satInf/modo)*100
        
    return ("Superior - {:.2f}%  Inferior - {:.2f}%".format(porcSatSup, porcSatInf))    

# Calcula a variabilidade dos dados
def variabilidade(data):
    
    #calculo dos quartis 
    upper_quartile = np.percentile(data, 75, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
    lower_quartile = np.percentile(data, 25, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)

    #estatisticas inter-quartis (iqr)
    iqr = upper_quartile - lower_quartile

    upper_whisker = data[data<=upper_quartile+1.5*iqr].max()
    lower_whisker = data[data>=lower_quartile-1.5*iqr].min()
    
    stdev_estimado = iqr/1.349
    variabilidade = stdev_estimado/np.average(data)
    
    return "{:.2f}%".format(variabilidade*100)

# Calcula o flatline
def flatlineSensor(data, rang):
    
    dataStd = data.rolling(60).std()
    
    varStd = rang/10**6
        
    flatline = (dataStd[(dataStd > -varStd) & (dataStd < varStd)].count()/dataStd.shape[0])*100
      
    return "{:.2f}%".format(flatline)