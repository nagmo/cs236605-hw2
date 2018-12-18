from hw2 import experiments

for L in [2, 4, 8]:
    for K in [32, 64, 128, 258]:
        experiments.run_experiment(f'exp1_1_K{K}_L{L}', layers_per_block=L, filters_per_layer=[K], epochs=30)

