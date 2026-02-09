
from time import time
from tcc_funcoes import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from xgboost import XGBRegressor

from itertools import product

dc_treino = abre_arquivo_pickle('dc_treino.pkl')
dc_nome_salvar = abre_arquivo_pickle('dc_nome_salvar.pkl')

####################################################################################################
############################################# XGBoost ##############################################
####################################################################################################

### https://xgboost.readthedocs.io/en/stable/python/python_api.html

print(" XGBoost ".center(100, "="))
print()

grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [6, 8, 10],
    'min_child_weight': [1, 5, 10],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'subsample': [0.7, 0.85, 1.0]
}

parametros_fixos = {
    'n_estimators': 100,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse'
}

keys, values = zip(*grid.items())

combinacoes = [dict(zip(keys, v)) for v in product(*values)]

del grid, keys, values
gc.collect()

dc_xgb_grid_search = {}

for nome, df in dc_treino.items():

    dc_xgb_grid_search[nome] = {}

    print(f" {nome} ".center(100, "-"))
    print()

    print(f"{len(combinacoes)} combinações de hiperparâmetros")
    print()

    x_treino = dc_treino[nome]['x'].astype('float32')
    y_treino = dc_treino[nome]['y'].astype('float32')

    x_treino, x_validacao, y_treino, y_validacao = train_test_split(
        x_treino,
        y_treino,
        test_size=1/8,
            # x_validacao terá 10% dos dados originais
            # x_treino terá 70% dos dados originais
            # x_teste terá 20% dos dados originais
        random_state=42,
        shuffle=True,
    )

    gc.collect()

    nome_salvar = dc_nome_salvar[nome]

    for i, parametro in enumerate(combinacoes, 1):

        tempo_inicio = time()

        print(f"{i:2} / {len(combinacoes):2} | {parametro}", end=" | ", flush=True)

        try:

            xgb = XGBRegressor(
                random_state=42,
                verbosity=0,
                **parametro,
                **parametros_fixos,
            )

            xgb.fit(x_treino, y_treino)

            z_validacao = xgb.predict(x_validacao)

            rmse = root_mean_squared_error(y_validacao.to_numpy(), z_validacao)

            del xgb
            gc.collect()

            tempo_fim = time()

            print(f"RMSE: {rmse:.4f} | Tempo: {tempo_fim - tempo_inicio:.1f}s")

            dc_xgb_grid_search[nome][str(parametro)] = {
                'rmse': rmse,
                'tempo': tempo_fim - tempo_inicio,
                'parametro': parametro,
            }
            print()

        except Exception as e:

            print()
            print()
            print('#' * 100)
            print('#' * 100)
            print(f'Erro na combinação {parametro} | {e}'.center(100))
            print('#' * 100)
            print('#' * 100)
            print()
            print()

            if 'xgb' in locals():
                del xgb

            gc.collect()

    salva_arquivo_pickle(f'{nome_salvar}_xgb_grid_search.pkl', dc_xgb_grid_search[nome])

salva_arquivo_pickle('dc_xgb_grid_search.pkl', dc_xgb_grid_search)