from .hw2 import experiments

for K in [32, 64]:
    for L in [2, 4, 8, 16]:
        experiments.run_experiment(f'exp1_1_K{K}_L{L}', layers_per_block=L, filters_per_layer=[K])
