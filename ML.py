import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import numpy as np


# Pré-processamento
dataset['Comentário'] = dataset['Qual seu nome?'].astype(str).fillna('')

# Analise de Sentimentos
def analisar_sentimento(texto):
    analise = TextBlob(texto)
    return analise.sentiment.polarity  # Escala de -1 (negativo) a 1 (positivo)

dataset['Sentimento'] = dataset['Comentário'].apply(analisar_sentimento)

# Extração de palavras mais frequentes
vectorizer = CountVectorizer(stop_words='portuguese', max_features=10)
frequencias = vectorizer.fit_transform(dataset['Comentário'])

termos_frequentes = pd.DataFrame(
    frequencias.toarray(), columns=vectorizer.get_feature_names_out()
).sum().sort_values(ascending=False)

# Converte para string 
top_10_palavras = ', '.join(termos_frequentes.index)
dataset['Top_Palavras'] = top_10_palavras

# Distribuição das Notas
questoes = [
    'Você gosto do conteúdo do seu curso?',
    'Como você avalia o conhecimento do professor?',
    'É fácil se realizar as atividades das aulas?',
    'Como você avaliaria os métodos de avaliação?',
    'As instalações da escola são bem conservadas?',
    'A equipe de vendas conseguir esclarecer como seria o curso ?'
]

distribuicoes = {}
for questao in questoes:
    distribuicoes[questao] = dataset[questao].value_counts().sort_index()

for questao, distrib in distribuicoes.items():
    dataset[f'{questao}_Distribuicao'] = distrib.reindex(range(1, 6), fill_value=0).values

# Correlação entre Respostas e Sentimento
for questao in questoes:
    dataset[questao] = pd.to_numeric(dataset[questao], errors='coerce')

# Calcula a correlação entre as respostas e o sentimento
correlacoes = dataset[questoes].corrwith(dataset['Sentimento'])

# adiciona correlações ao dataset como uma string formatada
dataset['Correlação_Respostas_Sentimento'] = ', '.join(
    [f'{questao}: {correlacao:.2f}' for questao, correlacao in zip(questoes, correlacoes)]
)

# dataset final para Power BI
dataset
