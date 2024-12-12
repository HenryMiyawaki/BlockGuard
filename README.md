# Detecção de Ataques em Blockchain com Aprendizado de Máquina

![](https://img.shields.io/badge/Code-Python-informational?style=for-the-badge&logo=python&logoColor=white&color=BD2A95)

## 1️⃣ Introdução

As redes e sistemas descentralizados como blockchain, têm ganhado destaque como uma alternativa eficiente e inovadora em um mundo cada vez mais preocupado com privacidade e autonomia ao se destacarem por: 

> - Eliminar a dependência de uma entidade central
> - Colaboratividade e Escalabilidade da rede
> - Segurança e Imutabilidade

## 2️⃣ Objetivos

Propomos, assim, um modelo que com os seguintes objetivos

> - ✅ Identifique tipos de ataques comuns em Blockchain.
>   - Brute Force
>   - Denial of Serivice (Dos)
>   - Flooding of Transactions
>   - Man-in-the-Middle
> - ✅ O modelo possa aprender com outros nós da rede.
> - ✅ Aprendizado descentralizada sem expor os dados dos peers.

## 3️⃣ Metodologia 

Foi delimitado nossa problematica como um problema de **Classificação** onde seria necessario a partir de variaveis independentes classificar uma certa requisição em 4 categorias.

### 📈 Dataset

O dataset escolhido foi o BNaT Dataset (Blockchain Network Attack Traffic Dataset) uma base de dados coletados de nós **Ethereum** que contêm os ataques listados nos objetivos.

-> [Conheça o BNaT Dataset](https://paperswithcode.com/dataset/bnat)

### 💻 Modelo

Para resolver o problema de classificação foi escolhido criar um modelo de **Rede Neural** devido a quantidade de itens para serem classificados, flexibilidade que uma rede neural pode ofererecer.

> Foi utilizado as seguintes ferramentas para a criação da rede: 
> - Python 3.12.4
> - Pytorch
> - Scikit-learn
> - Numpy
> - Pandas

### 📊 Pré-processamento

> Para melhorar a qualidade e divisão dos dados, foi adotada as seguintes técnicas: 
> - Estratificação para a separação dos conjuntos de treino e teste
> - **Transformação** matrizes de mútiplas dimensões para **vectorized operation**
> - Carregamento dos dados em lotes (batches embaralhados de 32 amostras)

### Rede Neural

### Knowledge Distillation

## Treinamento

## Peer Network

## Vantagens do Peer Network

## Resultados
