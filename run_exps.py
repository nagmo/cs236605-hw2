from hw2 import experiments
import traceback
from hw2 import models
import torch

# for K in [32, 64]:
#     for L in [2, 4, 8, 16]:
#         try:
#             experiments.run_experiment(f'exp1_1_K{K}_L{L}', layers_per_block=L, filters_per_layer=[K], epochs=100)
#         except:
#             traceback.print_exc()
#
# for L in [2, 4, 8]:
#     for K in [32, 64, 128, 258]:
#         try:
#             experiments.run_experiment(f'exp1_2_L{L}_K{K}', layers_per_block=L, filters_per_layer=[K], epochs=100)
#         except:
#             traceback.print_exc()
#
# for L in [1, 2, 3, 4]:
#     try:
#         experiments.run_experiment(f'exp1_3_L{L}_K64-128-256', layers_per_block=L, filters_per_layer=[64, 128, 256], epochs=100)
#     except:
#         traceback.print_exc()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = models.YourCodeNet((3, 100, 100), 10, filters=[32]*4, pool_every=2, hidden_dims=[100]*2).to(device)
# print(net)
#
# test_image = torch.randint(low=0, high=256, size=(3, 100, 100)).to(device)
# test_out = net(test_image.unsqueeze(0))
# print('out =', test_out)

for K in [[64, 128, 256, 512]]:
    for L in [4]:
        try:
            experiments.run_experiment(f'exp2_L{L}_K64-128-256-512', layers_per_block=L, filters_per_layer=K, epochs=100, ycn=True)
        except:
            traceback.print_exc()
