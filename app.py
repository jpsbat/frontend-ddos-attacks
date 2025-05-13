import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

from utils import obter_colunas, preparar_dados

def calcular_status_seguranca(metricas):
    valores = list(metricas.values())
    media = np.mean(valores)
    
    if 0.90 <= media <= 1.0:
        return media, "Ataque detectado", "red"
    elif 0.60 <= media < 0.90:
        return media, "Movimentação suspeita detectada", "orange"
    elif 0.0 <= media < 0.60:
        return media, "Ambiente aparentemente normal", "green"
    else:
        return media, "⚠️ Valor de métricas fora do esperado", "yellow"

st.title("Detecção Inteligente de Ataques DDoS Utilizando Aprendizado de Máquina")
st.markdown("""
Este dashboard interativo foi desenvolvido para demonstrar a detecção de ataques de negação de serviço distribuído utilizando Machine Learning. O objetivo é apresentar um pipeline completo que inclui pré-processamento de dados, treinamento de modelos e avaliação de desempenho.
baseado no projeto de Trabalho de Conclusão de Curso.
""")

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Formato de arquivo não suportado. Por favor, carregue um arquivo .parquet ou .csv")
                return None, None, None, None
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")
            return None, None, None, None

        st.subheader("Visualização Inicial dos Dados")
        st.write(f"Shape do dataset: {df.shape}")
        st.write("Primeiras 5 linhas do dataset:", df.head())

        try:
            X, y, le, num_classes, col_valor_unico, col_corr_alta = preparar_dados(df)
        except Exception as e:
            st.error(str(e))
            return None, None, None, None

        if col_valor_unico:
            st.write(f"Colunas removidas por valor único: {col_valor_unico}")
        if col_corr_alta:
            st.write(f"Colunas removidas por alta correlação: {col_corr_alta}")

        st.write("Contagem de valores para 'Label' após codificação:", y.value_counts())
        st.write("Features numéricas normalizadas com MinMaxScaler.")

        col_cat, col_num, cat_alta_card = obter_colunas(df)
        st.write(f"Colunas categóricas: {col_cat}")
        st.write(f"Colunas numéricas: {col_num}")
        st.write(f"Colunas categóricas com alta cardinalidade: {cat_alta_card}")

        return X, y, le, num_classes
    return None, None, None, None

@st.cache_data
def load_and_preprocess_multiple_files(uploaded_files):
    if not uploaded_files:
        return None, None, None, None
    
    dfs = []
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                st.warning(f"Formato de arquivo não suportado: {uploaded_file.name}. Ignorando...")
                continue
                
            dfs.append(df)
            st.success(f"Arquivo carregado com sucesso: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Erro ao ler o arquivo {uploaded_file.name}: {e}")
    
    if not dfs:
        st.error("Nenhum arquivo válido foi carregado.")
        return None, None, None, None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    st.subheader("Visualização Inicial dos Dados Combinados")
    st.write(f"Shape do dataset combinado: {combined_df.shape}")
    st.write("Primeiras 5 linhas do dataset:", combined_df.head())
    
    try:
        X, y, le, num_classes, col_valor_unico, col_corr_alta = preparar_dados(combined_df)
    except Exception as e:
        st.error(str(e))
        return None, None, None, None
    
    if col_valor_unico:
        st.write(f"Colunas removidas por valor único: {col_valor_unico}")
    if col_corr_alta:
        st.write(f"Colunas removidas por alta correlação: {col_corr_alta}")
    
    st.write("Contagem de valores para 'Label' após codificação:", y.value_counts())
    st.write("Features numéricas normalizadas com MinMaxScaler.")
    
    col_cat, col_num, cat_alta_card = obter_colunas(combined_df)
    st.write(f"Colunas categóricas: {col_cat}")
    st.write(f"Colunas numéricas: {col_num}")
    st.write(f"Colunas categóricas com alta cardinalidade: {cat_alta_card}")
    
    return X, y, le, num_classes

@st.cache_resource
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def train_neural_network(X_train, y_train, num_classes, input_dim):
    y_train_categorical = to_categorical(y_train, num_classes=num_classes)
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    with st.spinner("Treinando a Rede Neural... Este processo pode ser demorado."):
        model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, verbose=0)
    return model

def evaluate_model(model, X_test, y_test, le, model_type="sklearn"):
    if model_type == "keras":
        y_pred_proba = model.predict(X_test)
        preds = np.argmax(y_pred_proba, axis=1)
    else:
        preds = model.predict(X_test)
        
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

    st.subheader("Métricas de Avaliação do Modelo")
    st.write(f"Acurácia: {accuracy:.4f}")
    st.write(f"Precisão: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1-Score: {f1:.4f}")
    
    metricas = {
        'acuracia': accuracy,
        'precisao': precision,
        'recall': recall,
        'f1': f1
    }
    
    media, status, cor = calcular_status_seguranca(metricas)
    
    st.subheader("Análise de Segurança")
    st.markdown(f"<div style='background-color: {cor}; padding: 10px; border-radius: 5px;'><h3 style='color: white; margin: 0;'>{status}</h3><p style='color: white; margin: 0;'>Média das métricas: {media:.4f}</p></div>", unsafe_allow_html=True)

    st.subheader("Relatório de Classificação")
    try:
        target_names = [str(cls) for cls in le.classes_]
        report = classification_report(y_test, preds, target_names=target_names, zero_division=0)
        st.text(report)
    except Exception as e:
        st.warning(f"Não foi possível gerar o relatório de classificação com nomes de classes: {e}. Exibindo com índices numéricos.")
        report = classification_report(y_test, preds, zero_division=0)
        st.text(report)

    st.subheader("Matriz de Confusão")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    if 'target_names' in locals() and len(target_names) == cm.shape[0]:
        ax.set_xticklabels(target_names)
        ax.set_yticklabels(target_names)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    st.pyplot(fig)
    
    return metricas

st.sidebar.header("Configurações do Detector de DDoS")

st.sidebar.subheader("1. Carregar Dados")
uploaded_files = st.sidebar.file_uploader("Escolha um ou mais arquivos Parquet ou CSV", type=["parquet", "csv"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"Total de arquivos carregados: {len(uploaded_files)}")
    X, y, le, num_classes = load_and_preprocess_multiple_files(uploaded_files)

    if X is not None and y is not None and num_classes is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        st.write(f"Dados divididos em treino ({X_train.shape[0]} amostras) e teste ({X_test.shape[0]} amostras).")
        st.write(f"Número de classes detectadas: {num_classes}")
        st.write(f"Dimensão dos dados de entrada (features): {X_train.shape[1]}")

        st.sidebar.subheader("2. Selecionar Modelo")
        model_choice = st.sidebar.selectbox("Escolha o modelo:", ["RandomForest", "Rede Neural (Keras)"])

        if st.sidebar.button("Treinar e Avaliar Modelo"):
            if model_choice == "RandomForest":
                with st.spinner("Treinando o modelo RandomForest..."):
                    model = train_random_forest(X_train, y_train)
                st.success("Modelo RandomForest treinado com sucesso!")
                metricas = evaluate_model(model, X_test, y_test, le, model_type="sklearn")
            elif model_choice == "Rede Neural (Keras)":
                if num_classes <= 1:
                    st.error("Para a Rede Neural, o número de classes deve ser maior que 1. Verifique a coluna 'Label' do seu dataset.")
                else:
                    X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
                    X_test_np = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
                    y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
                    y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

                    model = train_neural_network(X_train_np, y_train_np, num_classes, X_train_np.shape[1])
                    st.success("Modelo Rede Neural treinado com sucesso!")
                    metricas = evaluate_model(model, X_test_np, y_test_np, le, model_type="keras")
            else:
                st.error("Modelo não reconhecido.")
else:
    st.info("Por favor, carregue um ou mais arquivos de dados para começar.")

st.sidebar.markdown("---")
st.sidebar.markdown("Detecção Inteligente de Ataques DDoS Utilizando Aprendizado de Máquina")

from ddos_model import exibir_contexto_tecnico
exibir_contexto_tecnico()
