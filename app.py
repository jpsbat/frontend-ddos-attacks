import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Importações para Rede Neural
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

# st.set_option('deprecation.showPyplotGlobalUse', False) # Comentado: Opção obsoleta causa erro

# Título da Aplicação
st.title("Detecção Inteligente de Ataques DDoS com Machine Learning")
st.markdown("""
Este é um dashboard interativo para demonstrar a detecção de ataques DDoS utilizando modelos de Machine Learning,
baseado no projeto de TCC.
""")

# --- Funções de Carregamento e Pré-processamento (Adaptadas do Notebook) ---
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

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if 'Label' not in df.columns:
            st.error("A coluna 'Label' não foi encontrada no dataset. Esta coluna é necessária para o treinamento.")
            return None, None, None, None

        st.subheader("Pré-processamento dos Dados")
        le = LabelEncoder()
        df['Label_Encoded'] = le.fit_transform(df['Label'])
        st.write("Coluna 'Label' codificada.")
        st.write("Contagem de valores para 'Label' após codificação:", df['Label_Encoded'].value_counts())
        
        num_classes = df['Label_Encoded'].nunique()

        X = df.drop(['Label', 'Label_Encoded'], axis=1)
        y = df['Label_Encoded']

        numeric_cols = X.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            st.error("Nenhuma coluna numérica encontrada para aplicar o MinMaxScaler.")
            return None, None, None, None
            
        scaler = MinMaxScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        st.write("Features numéricas normalizadas com MinMaxScaler.")

        return X, y, le, num_classes
    return None, None, None, None

# --- Funções de Treinamento e Avaliação ---
@st.cache_resource # Usar cache para modelos treinados
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def train_neural_network(X_train, y_train, num_classes, input_dim):
    # Converter y_train para one-hot encoding
    y_train_categorical = to_categorical(y_train, num_classes=num_classes)
    
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) # Camada de saída com softmax para classificação multiclasse
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Treinamento (simplificado para o dashboard)
    # Idealmente, X_val e y_val seriam passados para validation_data
    with st.spinner("Treinando a Rede Neural... Este processo pode ser demorado."):
        model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, verbose=0) # verbose=0 para não poluir o output do Streamlit
    return model

def evaluate_model(model, X_test, y_test, le, model_type="sklearn"):
    if model_type == "keras":
        # Para modelos Keras, a predição retorna probabilidades
        y_pred_proba = model.predict(X_test)
        preds = np.argmax(y_pred_proba, axis=1)
    else: # model_type == "sklearn"
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

    st.subheader("Relatório de Classificação")
    try:
        # Tentar obter os nomes das classes originais se o LabelEncoder foi usado para y
        # e se y_test ainda não está one-hot encoded
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

# --- Interface do Usuário Streamlit ---
st.sidebar.header("Configurações do Detector de DDoS")

# 1. Upload do arquivo de dados
st.sidebar.subheader("1. Carregar Dados")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo Parquet ou CSV", type=["parquet", "csv"])

if uploaded_file:
    st.info(f"Arquivo carregado: {uploaded_file.name}")
    X, y, le, num_classes = load_and_preprocess_data(uploaded_file)

    if X is not None and y is not None and num_classes is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        st.write(f"Dados divididos em treino ({X_train.shape[0]} amostras) e teste ({X_test.shape[0]} amostras).")
        st.write(f"Número de classes detectadas: {num_classes}")
        st.write(f"Dimensão dos dados de entrada (features): {X_train.shape[1]}")

        # 2. Seleção do Modelo
        st.sidebar.subheader("2. Selecionar Modelo")
        model_choice = st.sidebar.selectbox("Escolha o modelo:", ["RandomForest", "Rede Neural (Keras)"])

        # 3. Treinamento e Avaliação
        if st.sidebar.button("Treinar e Avaliar Modelo"):
            if model_choice == "RandomForest":
                with st.spinner("Treinando o modelo RandomForest..."):
                    model = train_random_forest(X_train, y_train)
                st.success("Modelo RandomForest treinado com sucesso!")
                evaluate_model(model, X_test, y_test, le, model_type="sklearn")
            elif model_choice == "Rede Neural (Keras)":
                if num_classes <= 1:
                    st.error("Para a Rede Neural, o número de classes deve ser maior que 1. Verifique a coluna 'Label' do seu dataset.")
                else:
                    # Assegurar que X_train e X_test sejam DataFrames para Keras ou convertê-los para NumPy arrays
                    # Keras geralmente funciona bem com NumPy arrays
                    X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
                    X_test_np = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
                    y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
                    y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

                    model = train_neural_network(X_train_np, y_train_np, num_classes, X_train_np.shape[1])
                    st.success("Modelo Rede Neural treinado com sucesso!")
                    evaluate_model(model, X_test_np, y_test_np, le, model_type="keras")
            else:
                st.error("Modelo não reconhecido.")
else:
    st.info("Por favor, carregue um arquivo de dados para começar.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido como parte do TCC - Detecção Inteligente de Ataques DDoS")

