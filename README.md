# Detecção Inteligente de Ataques DDoS Utilizando Aprendizado de Máquina

## Trabalho de Conclusão de Curso

**Autoria:** [João Pedro Santos Batista, Vinicius Gonçalves Angelo, Vinicius Ribeiro Silva]
**Orientação:** Prof. Dra. Andréia Damasio Leles
**Instituição:** Centro Universitário Facens

---

## 1. Introdução e Objetivo do Projeto

Este projeto de Trabalho de Conclusão de Curso (TCC) foca na **Detecção Inteligente de Ataques DDoS Utilizando Aprendizado de Máquina**. Ataques DDoS representam uma ameaça significativa à disponibilidade de serviços online, causando interrupções e prejuízos financeiros. O objetivo principal deste trabalho é desenvolver e apresentar um sistema capaz de identificar padrões característicos de tráfego de rede malicioso associado a ataques DDoS, diferenciando-os do tráfego legítimo.

Para alcançar este objetivo, foi desenvolvido um dashboard interativo utilizando a biblioteca Streamlit em Python. Este dashboard permite:

*   Carregar datasets de tráfego de rede (nos formatos Parquet ou CSV).
*   Realizar o pré-processamento dos dados, incluindo limpeza, codificação de rótulos e normalização.
*   Treinar e avaliar dois modelos de Machine Learning para a detecção de ataques:
    *   RandomForest Classifier
    *   Rede Neural Artificial (implementada com Keras/TensorFlow)
*   Visualizar as métricas de desempenho dos modelos, como acurácia, precisão, recall, F1-score, relatório de classificação e matriz de confusão.

## 2. Tecnologias Utilizadas

O desenvolvimento deste projeto envolveu o uso das seguintes tecnologias e bibliotecas Python:

*   **Python 3:** Linguagem de programação principal.
*   **Streamlit:** Framework para a criação do dashboard web interativo.
*   **Pandas:** Para manipulação e análise de dados tabulares.
*   **NumPy:** Para operações numéricas eficientes.
*   **Scikit-learn:** Para a implementação do modelo RandomForest, pré-processamento de dados (LabelEncoder, MinMaxScaler) e cálculo de métricas de avaliação.
*   **TensorFlow (com Keras API):** Para a construção, treinamento e avaliação do modelo de Rede Neural Artificial.
*   **Matplotlib e Seaborn:** Para a geração de gráficos e visualizações, como a matriz de confusão.
*   **PyArrow:** Para a leitura eficiente de arquivos no formato Parquet.
*   **Jupyter Notebook/Google Colab:** Utilizado para a exploração inicial dos dados, desenvolvimento e experimentação dos modelos de machine learning.
*   **Visual Studio Code (VS Code):** Ambiente de desenvolvimento integrado (IDE) recomendado para edição do código.
*   **Git e GitHub (Recomendado):** Para versionamento de código e colaboração (não abordado diretamente, mas uma boa prática).

## 3. Dataset

O modelo foi desenvolvido e testado primariamente com o dataset **CICDDoS2019**, que é um conjunto de dados público e amplamente utilizado para pesquisa em detecção de intrusão e ataques DDoS. Este dataset contém uma variedade de ataques DDoS modernos e tráfego benigno.

*   **Fonte Original (Referência):** Canadian Institute for Cybersecurity (CIC) - University of New Brunswick.
*   **Disponibilidade:** Pode ser encontrado em plataformas como Kaggle (ex: [https://www.kaggle.com/datasets/dhoogla/cicddos2019](https://www.kaggle.com/datasets/dhoogla/cicddos2019)).

O dashboard permite o upload de arquivos de dados nos formatos `.parquet` ou `.csv`, possibilitando o uso do CICDDoS2019 ou outros datasets com estrutura similar (contendo features de tráfego de rede e uma coluna `Label` indicando se o tráfego é benigno ou um tipo de ataque).

## 4. Funcionalidades do Dashboard

O dashboard interativo (`app.py`) oferece as seguintes funcionalidades:

1.  **Upload de Dados:** Permite ao usuário carregar seu próprio dataset (formato Parquet ou CSV).
2.  **Visualização Inicial dos Dados:** Exibe informações básicas do dataset carregado, como o número de amostras e features, e as primeiras linhas.
3.  **Pré-processamento Automatizado:** Realiza a codificação da coluna alvo (`Label`) e a normalização das features numéricas utilizando MinMaxScaler.
4.  **Divisão dos Dados:** Separa o dataset em conjuntos de treinamento e teste.
5.  **Seleção de Modelo:** O usuário pode escolher entre treinar um modelo RandomForest ou uma Rede Neural (Keras).
6.  **Treinamento do Modelo:** Executa o treinamento do modelo selecionado com os dados carregados.
7.  **Avaliação do Modelo:** Apresenta métricas de desempenho detalhadas:
    *   Acurácia
    *   Precisão (ponderada)
    *   Recall (ponderado)
    *   F1-Score (ponderado)
    *   Relatório de Classificação (com métricas por classe)
    *   Matriz de Confusão visualizada.

## 5. Estrutura do Projeto

A estrutura de pastas e arquivos do projeto é:

```
ddos-attacks/       # Diretório raiz do projeto
├── app.py                     # Código principal da aplicação
├── ddos_pred.py                     # Código do notebook
├── utils.py                     # Funções importadas do notebook
├── requirements.txt           # Arquivo com as dependências do Python
```

O arquivo `DDOS_Pred (1).ipynb` (notebook Jupyter original) serve como base para a lógica de processamento e modelagem implementada no `app.py`.

## 6. Configuração e Execução

Instruções detalhadas para configurar o ambiente, instalar as dependências e executar o dashboard estão disponíveis no arquivo `GUIA_USUARIO.md`.

Resumidamente, os passos são:

1.  **Pré-requisitos:** Python 3.8+ e pip3 instalados.
2.  **Clonar/Baixar o Projeto:** Obtenha os arquivos do projeto.
3.  **Criar Ambiente Virtual (Recomendado):**
    ```bash
    cd ddos-attacks
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Instalar Dependências:**
    ```bash
    pip3 install streamlit pandas scikit-learn matplotlib seaborn pyarrow tensorflow
    ```
    (Ou, se um `requirements.txt` for fornecido: `pip3 install -r requirements.txt`)
5.  **Executar o Dashboard:**
    ```bash
    streamlit run app.py
    ```
6.  Acessar o dashboard no navegador através do URL fornecido.

## 7. Modelos Implementados

### a. RandomForest Classifier

*   Um modelo de ensemble learning baseado em árvores de decisão.
*   Implementado utilizando a biblioteca `scikit-learn`.
*   Conhecido por sua robustez e bom desempenho em diversas tarefas de classificação.

### b. Rede Neural Artificial (Keras/TensorFlow)

*   Um modelo de deep learning composto por camadas densas (fully connected layers).
*   Implementado utilizando a API Keras do TensorFlow.
*   A arquitetura base no `app.py` inclui:
    *   Camada de entrada com ativação ReLU.
    *   Uma camada oculta com ativação ReLU.
    *   Camada de saída com ativação Softmax (para classificação multiclasse).
*   Utiliza o otimizador Adam e a função de perda `categorical_crossentropy`.
*   Os dados da variável alvo (`Label`) são convertidos para o formato one-hot encoding para o treinamento da rede neural.
