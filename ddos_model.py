### Novo arquivo: ddos_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import streamlit as st

def treinar_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model

def treinar_rede_neural(X_train, y_train, num_classes, input_dim):
    from tensorflow.keras.utils import to_categorical # type: ignore
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=10, batch_size=32, verbose=0)
    return model

def avaliar_modelo(modelo, X_test, y_test, model_type="sklearn"):
    if model_type == "keras":
        y_pred_proba = modelo.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = modelo.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        'acuracia': acuracia,
        'precisao': precisao,
        'recall': recall,
        'f1': f1
    }

def exibir_contexto_tecnico():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sobre o Projeto")
    st.sidebar.info("""
    Este dashboard foi desenvolvido como parte do TCC para demonstrar a aplicação de algoritmos de Machine Learning na detecção de ataques DDoS.

    Os dados utilizados são do conjunto CIC-DDoS2019. O pipeline de pré-processamento, modelagem e avaliação foi originalmente desenvolvido e testado em um ambiente Jupyter Notebook (Google Colab), e adaptado aqui para uma aplicação interativa com Streamlit.
    """)

