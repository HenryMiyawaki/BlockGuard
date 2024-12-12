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
>   - Denial of Service (Dos)
>   - Flooding of Transactions
>   - Man-in-the-Middle
> - ✅ O modelo pode aprender com outros nós da rede.
> - ✅ Aprendizado descentralizada sem expor os dados dos peers.

## 3️⃣ Metodologia 

Foi delimitado nossa problemática como um problema de **Classificação** onde seria necessário a partir de variáveis independentes classificar uma certa requisição em 4 categorias.

### 📈 Dataset

O dataset escolhido foi o BNaT Dataset (Blockchain Network Attack Traffic Dataset) uma base de dados coletados de nós **Ethereum** que contêm os ataques listados nos objetivos.

-> [Conheça o BNaT Dataset](https://paperswithcode.com/dataset/bnat)

### 💻 Modelo

Para resolver o problema de classificação foi escolhido criar um modelo de **Rede Neural** devido a quantidade de itens para serem classificados, flexibilidade que uma rede neural pode oferecer.

> Foi utilizado as seguintes ferramentas para a criação da rede: 
> - Python 3.12.4
> - Pytorch
> - Scikit-learn
> - Numpy
> - Pandas

### 📊 Pré-processamento

> Para melhorar a qualidade e divisão dos dados, foi adotada as seguintes técnicas: 
> - Estratificação para a separação dos conjuntos de treino e teste
> - **Transformação** matrizes de múltiplas dimensões para **vectorized operation**
> - Carregamento dos dados em lotes (batches embaralhados de 32 amostras)

### 🧠 Rede Neural

A rede neural foi construída da seguinte forma:

#### **3.1 Topologia**
> - A primeira camada é totalmente conectada (linear) e transforma a entrada de dimensão N em uma saída de 128 Neurônios.
> - A segunda camada também totalmente conectada porém reduz de 128 neurônios para 64.
> - A terceira camada totalmente conectada reduz a dimensão de 64 para 32 neurônios.
> - Por fim, a camada de saída reduz os 32 neurônios para o número de classes do problema de classificação.

Todas as camadas seguem os seguintes passos:
> 1. Os dados passam pela camada linear
> 2. Sofrem uma normalização em lotes
> 3. É aplicado a função de ativação ReLU (Rectified Linear Unit)

#### **3.2 Função de Perda e Otimizador**
```diff
- Para calcular a perda da rede foi utilizado o Cross-Entropy-Loss. 
+ Para otimizar o modelo foi escolhido o Adaptive Moment Estimation (ADAM).
+ Scheduler foi usado para ajustar dinamicamente a taxa de aprendizado.
- Caso 3 épocas consecutivas sem progresso é aplicado um fator de 0.5 na taxa de aprendizado.
```
***

| 💾 Taxa de aprendizado       | 📐Regularização   |
| -----------                  | -----------        |
| 0.001                        | **L1 (Lasso)**     |


## 4️⃣ Treinamento

> Para o processo de treinamento foi definido um mecanismo de **Early Stopping** que interrompe o treinamento caso o modelo não apresente melhorias consecutivas em um determinado número de épocas.

Durante o processo de treino funcionalidades específicas são ativadas usadas como **dropout** e **normalização**
1. -> Para cada iteração moveremos os dados do batch para o dispositivo para a CPU ou GPU (CUDA)
2. -> Calcula-se a saída do modelo e a perda associada
3. -> É **zerado** os gradientes acumulados
4. -> Utilizamos as perdas calculadas para calcular o gradiente dos parâmetros do modelo.
5. -> Calculamos a perda do lote e a perda média das épocas
6. -> Por fim é verificado a condição de **Early Stopping**

## 5️⃣ Peer Network
> O objetivo central do sistema de Peer Network é criar uma rede descentralizada onde cada nó (peer) possui um modelo local, que pode ser treinado e atualizado com base em dados sintéticos e interações com outros peers, o sistema é projetado para.
> - Promover um aprendizado colaborativo
> - Utilizar o conceito de **Knowledge Distillation**
>   - Cada peer pode atuar como **teacher** quanto como **student**
> - Aprimorar o desempenho do modelo de forma descentralizada sem depender de um servidor central (Federated Learning) 

### Passos da Metodologia do Peer-Network
> 1. Geração de Dados Sintéticos Locais
>    - Utilização de datasets aleatórios para cada peer.
> 3. Treinamento Local do Modelo
>    - Feito maneira de autônoma
>    - Durante essa fase os peers não compartilham dados
> 5. Destilação de Conhecimento (Knowledge Destilation)
>    - 👩🏿‍🏫 Teacher (Modelo mais complexo) fornece feedback para o modelo menos treinado
>    - 🧑🏻‍🎓 Student (Modelo menos complexo) aprende com as previsões e parâmetros do Teacher
> 7. Evolução e Atualização
>    - Após a atualização modelo é permitido que o conhecimento de propague pela rede
> 9. Validação de Teste Descentralizada
>    - Para garantir que a evolução dos modelos seja efetiva, cada peer realiza testes de validação de forma local
> 11. Ajuste de Parâmetros e Estratégias de Colaboração
>     - Os peers podem adotar estratégias como a seleção de modelos de Teacher com base no desempenho ou até mesmo na complexidade do modelo.
>     - Os peers podem decidir quando compartilhar seu modelo com outros peers, ou quando evoluir de forma isolada.


## 6️⃣ Vantagens do Peer Network
As seguintes vantagens podem ser alcançadas com o Peer-Network sendo elas:
```diff
+ Descentralização com o sistema operando de forma distribuída.

+ Aprendizado colaborativo ao se utilizar do conceito de Teacher e Student.

+ Validação Contínua que permite que cada peer avalie e melhore o desempenho do seu modelo.

+ Evolução de conhecimento com base nas interações entre os peers permitindo a adaptabilidade a novos dados que os peers são expostos.
```

## 7️⃣ Resultados

Os resultados obtidos podem ser observados nos seguintes gráficos.

> ### ROC Curve
> <img src="/results/roc_curve.png" alt="Descrição da imagem" width="600">
> 
> ### Class Distribution
> <img src="/results/class_distribution.png" alt="Descrição da imagem" width="600">
> 
> ### Confusion Matrix
> <img src="/results/confusion_matrix.png" alt="Descrição da imagem" width="600">

# Contribuidores
- [Felipe Nunes Melo](https://github.com/felipemelonunes09)
- [Henry Miyawaki](https://github.com/HenryMiyawaki)
