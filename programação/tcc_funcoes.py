
from cudf import DataFrame as cudf_DataFrame
from typing import List, Tuple

import pandas as pd
import numpy as np
import gc

import matplotlib.pyplot as plt

####################################################################################################

def otimizar_memoria(df):

    '''
    Reduz o uso de memória convertendo tipos de dados
    '''

    from pandas.api.types import is_numeric_dtype

    for col in df.columns:

        col_type = df[col].dtype

        # Inteiro ou Float

        if is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Inteiros

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  

            # Floats

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

                else:
                    df[col] = df[col].astype(np.float64)

        # Object

        elif col_type == object:

            num_unique = len(df[col].unique())
            num_total = len(df[col])

            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    return df

####################################################################################################

def salva_arquivo_pickle(
    arquivo: str,
    variavel: object
) -> None:

    '''
    Salva uma variável em um arquivo pickle.
    '''

    from pickle import dump

    with open(arquivo, 'wb') as f:
        dump(variavel, f)

    return None

####################################################################################################

def abre_arquivos_pickle(
    v_arquivos_variavel: List[Tuple],
    contexto_global: dict
) -> None:

    '''
    Abre arquivos pickle e atribui o conteúdo a variáveis globais.
    '''

    from pickle import load

    for arquivo, variavel in v_arquivos_variavel:

        with open('dados/pickle/' + arquivo, 'rb') as f:
            contexto_global[variavel] = load(f)

    return None

####################################################################################################

def calcula_missing(
   df: pd.DataFrame | cudf_DataFrame
) -> pd.DataFrame:

    if type(df) == cudf_DataFrame:
        mask_nulos = (df.isnull().sum() > 0).to_pandas().values

    elif type(df) == pd.DataFrame:
        mask_nulos = (df.isnull().sum() > 0).values

    else:
        raise TypeError("O parâmetro 'df' deve ser do tipo pd.DataFrame ou cudf.DataFrame.")

    cols_missing = df.columns[mask_nulos]

    if len(cols_missing) > 0:

        cols_missing = (100 * df[cols_missing].isnull().sum() / df.shape[0])

        cols_missing = cols_missing.sort_values(ascending = False).reset_index()

        cols_missing.rename(columns = {0: '%_nulos', 'index': 'variavel'}, inplace = True)

        print("=" * 50)
        print()

        print("Percentual de valores ausentes por variável\n")

        print(cols_missing)

        print()
        print("=" * 50)

    else:
        print(f"Não há valores ausentes.")

    return cols_missing

####################################################################################################