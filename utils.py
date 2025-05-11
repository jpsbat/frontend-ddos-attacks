import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def obter_colunas(dataframe, lim_cat=10, lim_card=20):
    col_cat = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_comportando_cat = [col for col in dataframe.columns if dataframe[col].nunique() < lim_cat and dataframe[col].dtypes != "O"]
    cat_alta_card = [col for col in dataframe.columns if dataframe[col].nunique() > lim_card and dataframe[col].dtypes == "O"]

    col_cat = col_cat + num_comportando_cat
    col_cat = [col for col in col_cat if col not in cat_alta_card]

    col_num = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    col_num = [col for col in col_num if col not in num_comportando_cat]

    return col_cat, col_num, cat_alta_card

def remover_colunas_unicas(df):
    col_valor_unico = [col for col in df.columns if df[col].nunique() == 1]
    return df.drop(col_valor_unico, axis=1), col_valor_unico

def remover_colunas_correlacionadas(df, threshold=0.8):
    df_numerico = df.select_dtypes(include=[np.number])
    matriz_corr = df_numerico.corr().abs()
    mascara = np.triu(np.ones(matriz_corr.shape), k=1).astype(bool)
    tri_superior = matriz_corr.where(mascara)
    col_corr_alta = [col for col in tri_superior.columns if any(tri_superior[col] > threshold)]
    return df.drop(col_corr_alta, axis=1), col_corr_alta

def pipeline_processamento(df):
    # Remover duplicadas
    df = df.drop_duplicates()
    # Remover colunas com valor único
    df, col_valor_unico = remover_colunas_unicas(df)
    # Remover colunas altamente correlacionadas
    df, col_corr_alta = remover_colunas_correlacionadas(df)
    return df, col_valor_unico, col_corr_alta

def preparar_dados(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if 'Label' not in df.columns:
        raise ValueError("A coluna 'Label' não foi encontrada no dataset.")
    df, col_valor_unico, col_corr_alta = pipeline_processamento(df)
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['Label'])
    num_classes = df['Label_Encoded'].nunique()
    X = df.drop(['Label', 'Label_Encoded'], axis=1)
    y = df['Label_Encoded']
    numeric_cols = X.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return X, y, le, num_classes, col_valor_unico, col_corr_alta