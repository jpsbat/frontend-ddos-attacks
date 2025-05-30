# Detecção Inteligente de Ataques DDoS Utilizando Aprendizado de Máquina

## Trabalho de Conclusão de Curso

**Autoria:** João Pedro Santos Batista, Vinicius Gonçalves Angelo, Vinicius Ribeiro Silva  
**Orientação:** Prof. Dra. Andréia Damasio Leles  
**Instituição:** Centro Universitário Facens

---

## 1. Introdução e Objetivo do Projeto

Este projeto de Trabalho de Conclusão de Curso (TCC) foca na **Detecção Inteligente de Ataques DDoS Utilizando Aprendizado de Máquina**. Ataques DDoS representam uma ameaça significativa à disponibilidade de serviços online, causando interrupções e prejuízos financeiros. O objetivo principal deste trabalho é desenvolver e apresentar um sistema capaz de identificar padrões característicos de tráfego de rede malicioso associado a ataques DDoS, diferenciando-os do tráfego legítimo.

Para alcançar este objetivo, foi desenvolvido um dashboard interativo utilizando a biblioteca Streamlit em Python. Este dashboard permite:

* Carregar datasets de tráfego de rede (nos formatos Parquet ou CSV).
* Realizar análise exploratória completa dos dados carregados.
* Executar pré-processamento automatizado, incluindo limpeza, remoção de correlações altas, codificação de rótulos e normalização.
* Treinar e avaliar dois modelos de Machine Learning para a detecção de ataques:
  * RandomForest Classifier
  * Rede Neural (implementada com Keras/TensorFlow)
* Visualizar métricas de desempenho dos modelos, como acurácia, precisão, recall, F1-score e relatório de classificação detalhado.
* Apresentar análise de segurança baseada no desempenho dos modelos.

## 2. Tecnologias Utilizadas

O desenvolvimento deste projeto envolveu o uso das seguintes tecnologias e bibliotecas Python:

* **Python 3:** Linguagem de programação principal.
* **Streamlit:** Framework para a criação do dashboard web interativo.
* **Pandas:** Para manipulação e análise de dados tabulares.
* **NumPy:** Para operações numéricas eficientes.
* **Scikit-learn:** Para implementação do modelo RandomForest, pré-processamento de dados (LabelEncoder, MinMaxScaler) e cálculo de métricas de avaliação.
* **TensorFlow (com Keras API):** Para construção, treinamento e avaliação do modelo de Rede Neural Artificial.
* **Matplotlib e Seaborn:** Para geração de gráficos e visualizações.
* **PyArrow:** Para leitura eficiente de arquivos no formato Parquet.
* **Visual Studio Code (VS Code):** Ambiente de desenvolvimento integrado (IDE) utilizado para edição do código.

## 3. Dataset

O modelo foi desenvolvido e testado primariamente com o dataset **CICDDoS2019**, que é um conjunto de dados público e amplamente utilizado para pesquisa em detecção de intrusão e ataques DDoS. Este dataset contém uma variedade de ataques DDoS modernos e tráfego benigno.

* **Fonte Original:** Canadian Institute for Cybersecurity (CIC) - University of New Brunswick.
* **Disponibilidade:** Pode ser encontrado em plataformas como Kaggle.

O dashboard permite o upload de múltiplos arquivos de dados nos formatos `.parquet` ou `.csv`, possibilitando o uso do CICDDoS2019 ou outros datasets com estrutura similar (contendo features de tráfego de rede e uma coluna `Label` indicando se o tráfego é benigno ou um tipo de ataque).

## 4. Funcionalidades do Dashboard

O dashboard interativo oferece as seguintes funcionalidades organizadas em seções:

### Análise Exploratória dos Dados
1. **Informações Básicas:** Exibe métricas fundamentais como total de registros, features, valores ausentes e duplicatas.
2. **Distribuição das Classes:** Visualiza a distribuição das classes de ataques no dataset original.
3. **Análise de Tipos de Dados:** Categoriza colunas em categóricas, numéricas e de alta cardinalidade.
4. **Estatísticas Descritivas:** Apresenta estatísticas detalhadas das features numéricas.
5. **Amostra dos Dados:** Mostra as primeiras linhas do dataset para inspeção visual.

### Pré-processamento dos Dados
1. **Limpeza Automatizada:** Remove registros duplicados e valores infinitos/ausentes.
2. **Remoção de Colunas:** Elimina colunas com valor único e alta correlação automaticamente.
3. **Codificação de Labels:** Aplica LabelEncoder na variável target.
4. **Normalização:** Utiliza MinMaxScaler nas features numéricas.
5. **Visualização Final:** Mostra a distribuição das classes após o processamento.

### Modelagem e Avaliação
1. **Divisão dos Dados:** Separa automaticamente em 70% treino e 30% teste.
2. **Seleção de Modelo:** Permite escolher entre RandomForest e Rede Neural.
3. **Treinamento:** Executa o treinamento com feedback visual de progresso.
4. **Métricas de Desempenho:** Apresenta acurácia, precisão, recall e F1-score.
5. **Análise de Segurança:** Interpreta as métricas em termos de detecção de ameaças.
6. **Relatório Detalhado:** Fornece relatório de classificação completo por classe.

## 5. Estrutura do Projeto

A estrutura de pastas e arquivos do projeto é:

```
ddos-attacks/                  # Diretório raiz do projeto
├── __pycache__/              # Cache do Python (gerado automaticamente)
├── datasets/                 # Pasta contendo todos os datasets
├── venv/                     # Ambiente virtual Python
├── app.py                    # Código principal da aplicação Streamlit
├── utils.py                  # Funções utilitárias para pré-processamento
├── requirements.txt          # Dependências do Python
├── README.md                 # Documentação do projeto
└── .gitignore               # Arquivos ignorados pelo Git
```

## 6. Configuração e Execução

### Pré-requisitos
* Python 3.8 ou superior
* pip3 instalado

### Passos para execução

1. **Clonar/Baixar o Projeto:**
   ```bash
   git clone <url-do-repositorio>
   cd ddos-attacks
   ```

2. **Criar Ambiente Virtual (Recomendado):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Instalar Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Executar o Dashboard:**
   ```bash
   streamlit run app.py
   ```

5. **Acessar o Dashboard:**
   Abra o navegador no endereço fornecido (geralmente http://localhost:8501)

## 7. Modelos Implementados

### RandomForest Classifier
* Modelo de ensemble learning baseado em múltiplas árvores de decisão.
* Implementado com scikit-learn.
* Configuração: 50 estimadores, profundidade máxima 10, mínimo 5 amostras por folha.
* Vantagens: Rápido, interpretável e robusto contra overfitting.

### Rede Neural Artificial (Keras/TensorFlow)
* Modelo de deep learning com arquitetura feedforward.
* Implementado com Keras/TensorFlow.
* Arquitetura:
  * Camada de entrada: 128 neurônios com ativação ReLU
  * Camada oculta: 64 neurônios com ativação ReLU
  * Camada de saída: Número de classes com ativação Softmax
* Configuração: Otimizador Adam, loss categorical_crossentropy, 10 épocas.
* Vantagens: Capacidade de capturar padrões não-lineares complexos.

## 8. Dependências

As principais dependências estão listadas no arquivo `requirements.txt`:

* streamlit
* pandas
* scikit-learn
* matplotlib
* seaborn
* pyarrow
* tensorflow
* numpy

## 9. Contribuições e Desenvolvimento

Este projeto foi desenvolvido para fins acadêmicos como parte do Trabalho de Conclusão de Curso no Centro Universitário Facens, e serve como demonstração prática da aplicação de Machine Learning na detecção de ataques DDoS. O código está estruturado de forma modular, facilitando futuras extensões e melhorias.

