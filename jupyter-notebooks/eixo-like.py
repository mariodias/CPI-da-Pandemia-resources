"""este métdodo verifica a correlação entre os dados quantitativos da nossa base de dados.
Este é o primeiro passo para a composição de um indicador através da técnica de PCA."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
