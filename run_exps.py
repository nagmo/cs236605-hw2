from hw2 import experiments
import traceback

for K in [32, 64]:
    for L in [2, 4, 8, 16]:
        try:
            experiments.run_experiment(f'exp1_1_K{K}_L{L}', layers_per_block=L, filters_per_layer=[K], epochs=100)
        except:
            traceback.print_exc()

for L in [2, 4, 8]:
    for K in [32, 64, 128, 258]:
        try:
            experiments.run_experiment(f'exp1_2_L{L}_K{K}', layers_per_block=L, filters_per_layer=[K], epochs=100)
        except:
            traceback.print_exc()

for L in [1, 2, 3, 4]:
    try:
        experiments.run_experiment(f'exp1_3_L{L}_K64-128-256', layers_per_block=L, filters_per_layer=[64, 128, 256], epochs=100)
    except:
        traceback.print_exc()
