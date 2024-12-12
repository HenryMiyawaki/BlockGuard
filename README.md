# Detecção de Ataques em Blockchain com Aprendizado de Máquina

Status: ✅ Completo

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

### 🧠 Rede Neural

A rede neural foi construida da seguinte forma:

**3.1 Topologia**
> - A primeira camada é totalmente conectada (linear) transforma a entrada de dimensão N em uma sáida de 128 Neurônios.
> - A segunda camada também totalmente conectada porém reduz de 128 neurônios para 64.
> - A terceira camada totalmente conectada reduz a dimensão de 64 para 32 neurônios.
> - Por fim, a camada de saída reduz os 32 neurônios para o número de classes do problema de classificação.

Todas as camadas seguem os seguintes passos:
> 1. Os dados passame pela camada linear
> 2. Sofrem uma normalização em lotes
> 3. É aplicado a função de ativação ReLU (Rectified Linear Unit)

**3.2 Função de Perda e Otimizador**
```diff
- Para calcular a perda da rede foi utilizado o Cross-Entropy-Loss. 
+ Para otimizar o modelo foi escolhido o Adaptive Moment Estimation (ADAM).
+ Scheduler foi usado para ajsutar dinamicamente a taxa de aprendizado.
- Caso 3 épocas consecutivas sem progresso é aplicado um fator de 0.5 na taxa de aprendizado.
```
***

| 💾 Taxa de aprendizado       | 📐Regularização   |
| -----------                  | -----------        |
| 0.001                        | **L1 (Lasso)**     |


## 4️⃣ Treinamento

> Para o processo de treinamento foi definido um mecanismo de **Early Stopping** que interrome o treinamento caso o demolo não apresente melhorias consecutivas em um determiando número de épocas.

Durante o processo de treino funcionalidades especificas sáo atividas usadas como **dropout** e **normalização**
1. Para cada iteração, moveremos os dados do batch 

## Peer Network

## Vantagens do Peer Network

## Resultados

# Contribuidores
- [Felipe Nunes Melo](https://github.com/felipemelonunes09)
- [Henry Miyawaki](https://github.com/HenryMiyawaki)
