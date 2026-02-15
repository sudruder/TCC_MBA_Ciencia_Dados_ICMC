from tcc_funcoes import *

# dc_parametros_xgb = abre_arquivo_pickle('dc_parametros_xgb.pkl')
# dc_parametros_lgbm = abre_arquivo_pickle('dc_parametros_lgbm.pkl')
dc_parametros_rf = abre_arquivo_pickle('dc_parametros_rf.pkl')
dc_treino = abre_arquivo_pickle('dc_treino.pkl')

# print(' Parâmetros do XGBoost '.center(70, '='))
# print()

# for nome in dc_parametros_xgb.keys():

#     dc_parametros_xgb[nome].pop('rmse', None)

#     dc_parametros_xgb[nome]['max_depth'] = int(dc_parametros_xgb[nome]['max_depth'])
#     dc_parametros_xgb[nome]['min_child_weight'] = int(dc_parametros_xgb[nome]['min_child_weight'])

#     print(f'{nome}: {dc_parametros_xgb[nome]}')

# print()


# print(' Parâmetros do LightGBM '.center(70, '='))
# print()

# for nome in dc_parametros_lgbm.keys():

#     dc_parametros_lgbm[nome].pop('rmse', None)

#     dc_parametros_lgbm[nome]['num_leaves'] = int(dc_parametros_lgbm[nome]['num_leaves'])
#     dc_parametros_lgbm[nome]['min_child_samples'] = int(dc_parametros_lgbm[nome]['min_child_samples'])

#     print(f'{nome}: {dc_parametros_lgbm[nome]}')

# print()

# print(' Parâmetros do Random Forest '.center(70, '='))
# print()

for nome in dc_parametros_rf.keys():

    dc_parametros_rf[nome].pop('rmse', None)

    dc_parametros_rf[nome]['max_depth'] = int(dc_parametros_rf[nome]['max_depth'])

#     print(f'{nome}: {dc_parametros_rf[nome]}')

# print()

# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from cuml.ensemble import RandomForestRegressor

# dc_xgb = {}
# dc_lgbm = {}
dc_rf = {}

# fixo_xgb = {
#     'objective': 'reg:squarederror',
#     'tree_method': 'hist',
#     'device': 'cuda',
#     'eval_metric': 'rmse',
#     'random_state': 42,
#     'verbosity': 0
# }

# fixo_lgbm = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'device': 'cpu',
#     'n_jobs': -1,
#     'random_state': 42,
#     'verbose': -1
# }

fixo_rf = {
    'split_criterion': 'mse',
    'bootstrap': True,
    'n_bins': 256,
    'min_samples_leaf': 15,
    'n_streams': 1,
    'random_state': 42,
    'verbose': 6
}

### XGBoost

# for nome, dc in dc_parametros_xgb.items():

#     dc_xgb[nome] = {}

#     dc_xgb[nome][100] = XGBRegressor(
#         **dc,
#         **fixo_xgb,
#         n_estimators=100
#     )

#     dc_xgb[nome][500] = XGBRegressor(
#         **dc,
#         **fixo_xgb,
#         n_estimators=500
#     )

#     dc_xgb[nome][1000] = XGBRegressor(
#         **dc,
#         **fixo_xgb,
#         n_estimators=1000
#     )

#     dc_xgb[nome][3000] = XGBRegressor(
#         **dc,
#         **fixo_xgb,
#         n_estimators=3000
#     )

### LightGBM

# for  nome, dc in dc_parametros_lgbm.items():

#     dc_lgbm[nome] = {}

#     dc_lgbm[nome][100] = LGBMRegressor(
#         **dc,
#         **fixo_lgbm,
#         n_estimators=100
#     )

#     dc_lgbm[nome][500] = LGBMRegressor(
#         **dc,
#         **fixo_lgbm,
#         n_estimators=500
#     )

#     dc_lgbm[nome][1000] = LGBMRegressor(
#         **dc,
#         **fixo_lgbm,
#         n_estimators=1000
#     )

#     dc_lgbm[nome][3000] = LGBMRegressor(
#         **dc,
#         **fixo_lgbm,
#         n_estimators=3000
#     )

### Random Forest

for nome, dc in dc_parametros_rf.items():

    dc_rf[nome] = {}

    # dc_rf[nome][100] = RandomForestRegressor(
    #     **dc,
    #     **fixo_rf,
    #     n_estimators=100
    # )

    # dc_rf[nome][500] = RandomForestRegressor(
    #     **dc,
    #     **fixo_rf,
    #     n_estimators=500
    # )

    dc_rf[nome][1000] = RandomForestRegressor(
        **dc,
        **fixo_rf,
        n_estimators=1000
    )

    # dc_rf[nome][3000] = RandomForestRegressor(
    #     **dc,
    #     **fixo_rf,
    #     n_estimators=3000
    # )

### XGBoost

# print(" XGBoost ".center(70, '='))
# print()

# for nome, dc in dc_xgb.items():

#     x_treino = dc_treino[nome]['x']
#     y_treino = dc_treino[nome]['y']

#     print(f" {nome} ".center(70, '-'))
#     print()

#     for n_estimators, modelo in dc.items():

#         print(f'{n_estimators} árvores...', end=' ', flush=True)

#         modelo.fit(x_treino, y_treino)

#     print('  feito!')
#     print()

# salva_arquivo_pickle('dc_modelos_xgb.pkl', dc_xgb)

# del nome, dc, x_treino, y_treino, n_estimators, modelo, dc_xgb
# gc.collect()

# ### LightGBM

# print(" LightGBM ".center(70, '='))
# print()

# for nome, dc in dc_lgbm.items():

#     x_treino = dc_treino[nome]['x']
#     y_treino = dc_treino[nome]['y']

#     print(f" {nome} ".center(70, '-'))
#     print()

#     for n_estimators, modelo in dc.items():

#         print(f'{n_estimators} árvores...', end=' ', flush=True)

#         modelo.fit(x_treino.to_numpy(), y_treino.to_numpy().ravel())

#     print('  feito!')
#     print()

# salva_arquivo_pickle('dc_modelos_lgbm.pkl', dc_lgbm)

# del nome, dc, x_treino, y_treino, n_estimators, modelo, dc_lgbm
# gc.collect()

### Random Forest

print(" Random Forest ".center(70, '='))
print()

for nome, dc in dc_rf.items():

    x_treino = dc_treino[nome]['x'].to_numpy().astype(np.float32)
    y_treino = dc_treino[nome]['y'].to_numpy().astype(np.float32)

    print(f" {nome} ".center(70, '-'))
    print()

    # if nome in ('Ciências Humanas', 'Ciências Natureza', 'Linguagem e Código', 'Matemática'):
    # if nome in ('Ciências Humanas', 'Ciências Natureza', 'Linguagem e Código'):
    # if nome in ('Ciências Humanas', 'Ciências Natureza'):
    # if nome in ('Ciências Humanas'):

        # print(f"Pulando {nome}...")
        # print()

        # continue

    for n_estimators, modelo in dc.items():

        print(f'{n_estimators} árvores...', end=' ', flush=True)

        modelo.fit(x_treino, y_treino)

        print('Salvando modelo...', end=' ', flush=True)
        salva_arquivo_pickle(f'modelo_rf_{nome}_{n_estimators}.pkl', modelo)

        del modelo, n_estimators
        gc.collect()

    del nome, dc, x_treino, y_treino
    gc.collect()

    print('  feito!')
    print()