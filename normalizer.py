import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Leia os dados do arquivo CSV
dados = pd.read_csv('poker-hand-testing.data', sep=',')

# Seleciona apenas as colunas que você deseja normalizar (features)
colunas_para_normalizar = dados.columns[:-1] # Seleciona todas as colunas exceto a última (supondo que ela seja a coluna das classes)

# Inicializa o objeto MinMaxScaler
scaler = MinMaxScaler()

# Normaliza os dados
dados_normalizados = dados.copy() # Faça uma cópia dos dados originais
dados_normalizados[colunas_para_normalizar] = scaler.fit_transform(dados[colunas_para_normalizar])

# Salva os dados normalizados em um novo arquivo
dados_normalizados.to_csv('dados_normalizados.csv', index=False) # index=False para não salvar o índice do DataFrame no arquivo