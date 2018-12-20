import os
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf
import hw2.training as training
import hw2.blocks as blocks
import hw2.models as models
import hw2.answers as answers
import hw2.optimizers as optimizers

seed = 42

test = unittest.TestCase()

data_dir = os.path.join(os.getenv('HOME'), '.pytorch-datasets')
ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())

batch_size = 50
max_batches = 100
in_features = 3*32*32
num_classes = 10
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size//2, shuffle=False)

def train_with_optimizer(opt_name, opt_class, fig):
    torch.manual_seed(seed)
    res = []
    # Get hyperparameters
    # hp = answers.part2_optim_hp()
    reg, lr_rmsprop, lr_momentum = np.logspace(0.01, 0.4), np.logspace(0.0001, 1), np.logspace(0.0001, 1)
    max = 0
    for momentum in lr_momentum:
        hp = dict(wstd=0.3, lr_vanilla=0.001, lr_momentum=momentum,
                    lr_rmsprop=lr_rmsprop[0], reg=reg)
        hidden_features = [128] * 5
        num_epochs = 10

        # Create model, loss and optimizer instances
        model = models.MLP(in_features, num_classes, hidden_features, wstd=hp['wstd'])
        loss_fn = blocks.CrossEntropyLoss()
        optimizer = opt_class(model.params(), learn_rate=hp[f'lr_{opt_name}'], reg=hp['reg'])

        # Train with the Trainer
        trainer = training.BlocksTrainer(model, loss_fn, optimizer)
        fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=max_batches)
        if fit_res > max:
            max = fit_res
            res.append((momentum, max))
    return fig

fig_optim = None
fig_optim = train_with_optimizer('momentum', optimizers.MomentumSGD, fig_optim)