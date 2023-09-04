"""Conjunto de métodos para a composição e validação de um indicador de pontos de engajamento de tweets.
   Para a criação do indicador, denominado como "Eixo-Like, utilizei o método de redução PCA.
   Os métodos abaixo estão na ordem de execução necessária para a criação do indicador."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.stats import bartlett
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# este métdodo verifica a correlação entre os dados quantitativos da nossa base de dados.
def plot_correlation_matrix():

    # importa o dataset e cria um novo dataset somente com as colunas likes_count, retweets_count, replies_count que serão utilizadas na composição do indicador.
    df = pd.read_csv('/content/df_tweets_cpi_pandemia.csv')
    df_variables = df.loc[:, ['likes_count', 'retweets_count', 'replies_count']]

    # cria a matriz de correlação
    corr_matrix = df_variables.corr().round(4)
    sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd', fmt='.5f')

    # exibe a matriz de correlação.
    plt.title('Matriz de correlação entre as variáveis')
    plt.text(0.5, -0.5, 'matplotlib', fontsize=12)
    plt.show()


""" este método calcula o KMO da nossa base. O KMO é uma medida de adequação de amostra usada na análise de componentes principais 
e na análise fatorial. Ele avalia se os dados são apropriados para essas técnicas estatísticas, indicando se a amostra de dados é
grande o suficiente e se os dados são correlacionados o suficiente para realizar uma análise significativa. 
Um valor KMO alto (geralmente acima de 0,7) indica que os dados são adequados para análise fatorial ou de componentes principais."""
def calcular_kmo(corr_matrix):
    
    # calcula a matriz de correlação entre as variáveis
    corr_matrix = np.array(corr_matrix)

    # calcula a matriz de correlação parcial
    correlacao_parcial = np.zeros_like(corr_matrix)
    n = corr_matrix.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            r = -corr_matrix[i,j]/np.sqrt(corr_matrix[i,i]*corr_matrix[j,j])
            correlacao_parcial[i,j] = r
            correlacao_parcial[j,i] = r

    # calcula os valores de comunalidade
    comunalidades = np.sum(corr_matrix**2, axis=1)

    # calcula a matriz de correlação residual
    correlacao_residual = np.diag(comunalidades) - corr_matrix

    # calcula a soma das correlações residuais ao quadrado
    soma_quadrados_residual = np.sum(correlacao_residual**2)

    # calcula a soma das correlações totais ao quadrado
    soma_quadrados_total = np.sum(corr_matrix**2)

    # calcula o índice de KMO
    kmo = soma_quadrados_total / (soma_quadrados_total + soma_quadrados_residual)

    # calcula o teste de Bartlett
    estatistica_bartlett, p_valor = bartlett(*corr_matrix)

    # calcula o número de variáveis
    n_variaveis = corr_matrix.shape[0]

    # calcula o número de parâmetros
    n_parametros = int(n_variaveis*(n_variaveis-1)/2)

    # calcula o número de itens comuns
    n_itens_comuns = n_variaveis - np.sum(comunalidades == 0)

    # calcula a matriz de correlação parcial invertida
    matriz_inversa = inv(correlacao_parcial)

    # calcula o índice de KMO corrigido
    kmo_corrigido = kmo / (1 - (1/n_itens_comuns))

    # imprime os resultados
    print(f"KMO = {kmo:.3f}")
    print(f"KMO Corrigido = {kmo_corrigido:.3f}")
    print(f"Teste de Bartlett: estatística = {estatistica_bartlett:.3f}, p-valor = {p_valor:.3f}")

    if kmo >= 0.7:
        print("Adequado para Análise de Componentes Principais")
    else:
        print("Inadequado para Análise de Componentes Principais")

""" este método calcula a variância acumulada. A variância acumulada é uma medida importante em análises de Componentes Principais 
(PCA - Principal Component Analysis). 
Ela desempenha um papel fundamental ao ajudar a determinar quantos componentes principais 
devem ser retidos em uma redução de dimensionalidade ou análise de fatores."""
def calcular_variancia_acumulada():
    
    # Padronize os dados
    scaler = StandardScaler()
    dados_twitter_padronizados = scaler.fit_transform(df_variables)

    # Realize a análise de componentes principais (PCA)
    pca = PCA()
    pca.fit(dados_twitter_padronizados)

    # Exiba a proporção de variância explicada por cada componente
    print(pca.explained_variance_ratio_)

    # Selecione o número de componentes principais desejados e transforme os dados
    pca = PCA(n_components=3)
    dados_twitter_principal = pca.fit_transform(dados_twitter_padronizados)

    # Exiba os dados transformados com as novas variáveis principais
    return(dados_twitter_principal)
