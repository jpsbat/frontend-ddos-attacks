import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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
        return media, "🔴 Ataque detectado!", "red"
    elif 0.60 <= media < 0.90:
        return media, "🟠 Movimentação suspeita detectada", "orange"
    elif 0.0 <= media < 0.60:
        return media, "🟢 Ambiente aparentemente normal", "green"
    else:
        return media, "⚠️ Valor de métricas fora do esperado", "yellow"

st.set_page_config(
    page_title="Detector DDoS ML", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Detecção Inteligente de Ataques DDoS Utilizando Aprendizado de Máquina")

# FUNÇÕES DE CARREGAMENTO E PRÉ-PROCESSAMENTO

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

        return df
    return None

@st.cache_data
def load_and_preprocess_multiple_files(uploaded_files):
    if not uploaded_files:
        return None
    
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
            st.success(f"✅ Arquivo carregado com sucesso: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"❌ Erro ao ler o arquivo {uploaded_file.name}: {e}")
    
    if not dfs:
        st.error("Nenhum arquivo válido foi carregado.")
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def exploratory_data_analysis(df):
    st.header("**1. Análise Exploratória dos Dados**")
    st.markdown("---")
    
    # Informações básicas do dataset
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Registros", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total de Features", df.shape[1])
    with col3:
        st.metric("Valores Ausentes", df.isnull().sum().sum())
    with col4:
        st.metric("Registros Duplicados", df.duplicated().sum())
    
    st.markdown("### **Distribuição das Classes (Labels)**")
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(label_counts.to_frame(), use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            label_counts.plot(kind='bar', ax=ax, color='skyblue')
            plt.title('Distribuição das Classes')
            plt.xlabel('Classes')
            plt.ylabel('Quantidade')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("### **Análise dos Tipos de Dados**")
    col_cat, col_num, cat_alta_card = obter_colunas(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**5 Primeiras Colunas Categóricas:** {len(col_cat)}")
        if col_cat:
            st.write(col_cat[:5])
    
    with col2:
        st.info(f"**5 Primeiras Colunas Numéricas:** {len(col_num)}")
        if col_num:
            st.write(col_num[:5])
    
    with col3:
        st.warning(f"**Alta Cardinalidade (Primeiras 5):** {len(cat_alta_card)}")
        if cat_alta_card:
            st.write(cat_alta_card[:5])
    
    st.markdown("### **Estatísticas Descritivas (Features Numéricas)**")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe(), use_container_width=True)
    
    st.markdown("### **Amostra dos Dados**")
    st.dataframe(df.head(10), use_container_width=True)

def preprocessing_section(df):
    st.header("**Pré-Processamento dos Dados**")
    st.markdown("---")
    
    try:
        X, y, le, num_classes, col_valor_unico, col_corr_alta = preparar_dados(df)
        
        # Informações sobre o pré-processamento
        col1, col2 = st.columns(2)
        
        with col1:
            if col_valor_unico:
                st.warning(f"**Colunas removidas (valor único):** {len(col_valor_unico)}")
                with st.expander("Ver colunas removidas"):
                    st.write(col_valor_unico)
            else:
                st.success("✅ Nenhuma coluna com valor único encontrada")
        
        with col2:
            if col_corr_alta:
                st.warning(f"🔗 **Colunas removidas (alta correlação):** {len(col_corr_alta)}")
                with st.expander("Ver colunas removidas"):
                    st.write(col_corr_alta)
            else:
                st.success("✅ Nenhuma coluna com alta correlação encontrada")
        
        st.success("✅ **Normalização aplicada**: MinMaxScaler nas features numéricas")
        st.success("✅ **Classes codificadas numericamente**: LabelEncoder aplicado na classe-alvo")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features Finais", X.shape[1])
        with col2:
            st.metric("Total de Amostras", X.shape[0])
        with col3:
            st.metric("Número de Classes", num_classes)
        
        st.markdown("### **Distribuição Final das Classes**")
        class_distribution = y.value_counts().sort_index()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(class_distribution.to_frame(), use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            class_distribution.plot(kind='bar', ax=ax, color='lightgreen')
            plt.title('Distribuição das Classes Após Pré-processamento')
            plt.xlabel('Classe Codificada')
            plt.ylabel('Quantidade')
            plt.tight_layout()
            st.pyplot(fig)
        
        return X, y, le, num_classes
        
    except Exception as e:
        st.error(f"❌ Erro durante o pré-processamento: {str(e)}")
        return None, None, None, None

# TREINAMENTO DE MODELOS

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

def modeling_section(X, y, le, num_classes):
    st.header("**Modelagem**")
    st.markdown("---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    st.markdown("### **Divisão dos Dados**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dados de Treino", f"{X_train.shape[0]:,}")
    with col2:
        st.metric("Dados de Teste", f"{X_test.shape[0]:,}")
    with col3:
        st.metric("Proporção Teste", "30%")
    
    return X_train, X_test, y_train, y_test

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

    st.markdown("### **Métricas de Desempenho**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Acurácia", f"{accuracy:.4f}")
    with col2:
        st.metric("Precisão", f"{precision:.4f}")
    with col3:
        st.metric("Recall", f"{recall:.4f}")
    with col4:
        st.metric("F1-Score", f"{f1:.4f}")
    
    metricas = {
        'acuracia': accuracy,
        'precisao': precision,
        'recall': recall,
        'f1': f1
    }
    
    media, status, cor = calcular_status_seguranca(metricas)
    
    st.markdown("### **Análise de Segurança**")
    st.markdown(f"""
    <div style='background-color: {cor}; padding: 20px; border-radius: 10px; border-left: 5px solid darkred;'>
        <h2 style='color: white; margin: 0; text-align: center;'>{status}</h2>
        <p style='color: white; margin: 10px 0 0 0; text-align: center; font-size: 18px;'>
            Média das métricas: {media:.4f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### **Relatório Detalhado de Classificação**")
    try:
        target_names = [str(cls) for cls in le.classes_]
        report = classification_report(y_test, preds, target_names=target_names, zero_division=0)
        st.text(report)
    except Exception as e:
        st.warning(f"⚠️ Não foi possível gerar o relatório com nomes de classes: {e}")
        report = classification_report(y_test, preds, zero_division=0)
        st.text(report)
    
    return metricas

# INTERFACE PRINCIPAL

st.sidebar.header("Configurações do Detector de DDoS")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Carregar Dados")
uploaded_files = st.sidebar.file_uploader(
    "Escolha um ou mais arquivos", 
    type=["parquet", "csv"], 
    accept_multiple_files=True,
    help="Carregue arquivos no formato Parquet ou CSV contendo dados de tráfego de rede"
)

if uploaded_files:
    st.info(f"**Total de arquivos carregados:** {len(uploaded_files)}")
    
    # Carregamento dos dados
    df = load_and_preprocess_multiple_files(uploaded_files)
    
    if df is not None:
        # 1. ANÁLISE EXPLORATÓRIA
        exploratory_data_analysis(df)
        
        # 2. PRÉ-PROCESSAMENTO
        X, y, le, num_classes = preprocessing_section(df)
        
        if X is not None and y is not None and num_classes is not None:
            # 3. MODELAGEM
            X_train, X_test, y_train, y_test = modeling_section(X, y, le, num_classes)
            
            st.sidebar.subheader("2. Selecionar Modelo")
            model_choice = st.sidebar.selectbox(
                "Escolha o algoritmo:", 
                ["RandomForest", "Rede Neural (Keras)"],
                help="RandomForest: Rápido e interpretável\nRede Neural: Mais complexo, pode captar padrões não-lineares"
            )
            
            if st.sidebar.button("Treinar e Avaliar Modelo", type="primary"):
                if model_choice == "RandomForest":
                    with st.spinner("Treinando o modelo RandomForest..."):
                        model = train_random_forest(X_train, y_train)
                    st.success("✅ Modelo RandomForest treinado com sucesso!")
                    metricas = evaluate_model(model, X_test, y_test, le, model_type="sklearn")
                    
                elif model_choice == "Rede Neural (Keras)":
                    if num_classes <= 1:
                        st.error("❌ Para a Rede Neural, o número de classes deve ser maior que 1. Verifique a coluna 'Label' do seu dataset.")
                    else:
                        X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
                        X_test_np = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
                        y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
                        y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

                        model = train_neural_network(X_train_np, y_train_np, num_classes, X_train_np.shape[1])
                        st.success("✅ Modelo Rede Neural treinado com sucesso!")
                        metricas = evaluate_model(model, X_test_np, y_test_np, le, model_type="keras")
                else:
                    st.error("❌ Modelo não reconhecido.")
else:
    st.info("**Por favor, carregue um ou mais arquivos de dados para começar a análise.**")

# SIDEBAR - INFORMAÇÕES ADICIONAIS

st.sidebar.markdown("---")
st.sidebar.markdown("### **Sobre o Projeto**")
st.sidebar.info("""
Este dashboard foi desenvolvido como parte do TCC para demonstrar a aplicação de algoritmos de Machine Learning na detecção de ataques DDoS.

Os dados utilizados são do conjunto CIC-DDoS2019. O pipeline de pré-processamento, modelagem e avaliação foi originalmente desenvolvido e testado em um ambiente Jupyter Notebook (Google Colab), e adaptado aqui para uma aplicação interativa com Streamlit.
""")

st.sidebar.markdown("### **Configurações Técnicas**")
st.sidebar.text("""
• Normalização: MinMaxScaler
• Validação: 70% treino / 30% teste
• Random Forest: 50 árvores
• Rede Neural: 2 camadas ocultas
• Métricas: Acurácia, Precisão, Recall, F1
""")