
from time import time
from tcc_funcoes import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from cuml.ensemble import RandomForestRegressor
from itertools import product

dc_treino = abre_arquivo_pickle('dc_treino.pkl')
dc_nome_salvar = abre_arquivo_pickle('dc_nome_salvar.pkl')

####################################################################################################
########################################## Random Forest ###########################################
####################################################################################################

### https://docs.rapids.ai/api/cuml/stable/api/#random-forest

print(" Random Forest ".center(100, "="))
print()

grid = {
    'max_depth': [10, 15, 20],
    'max_features': [0.7, 0.9, 1.0],
    'max_samples': [0.8, 0.9, 1.0],
}

parametros_fixos = {
    'split_criterion': 'mse',
    'bootstrap': True,
    'n_bins': 256,
    'min_samples_leaf': 15,
    'n_streams': 4,
    'n_estimators': 100
}

keys, values = zip(*grid.items())

combinacoes = [dict(zip(keys, v)) for v in product(*values)]

del grid, keys, values
gc.collect()

dc_rf_grid_search = {}

for nome, df in dc_treino.items():

    dc_rf_grid_search[nome] = {}

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

            rf = RandomForestRegressor(
                random_state=42,
                verbose=False,
                **parametro,
                **parametros_fixos,
            )

            rf.fit(x_treino, y_treino)

            z_validacao = rf.predict(x_validacao)

            rmse = root_mean_squared_error(y_validacao.to_numpy(), z_validacao.to_numpy())

            del rf
            gc.collect()

            tempo_fim = time()

            print(f"RMSE: {rmse:.4f} | Tempo: {tempo_fim - tempo_inicio:.1f}s")

            dc_rf_grid_search[nome][str(parametro)] = {
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

            if 'rf' in locals():
                del rf

            gc.collect()

    salva_arquivo_pickle(f'{nome_salvar}_rf_grid_search.pkl', dc_rf_grid_search[nome])

salva_arquivo_pickle('dc_rf_grid_search.pkl', dc_rf_grid_search)