r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0, 0.002511, -9.653
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.3, 0.0015, 0.0001, 0.0000001, 0.01
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr = 0.5, 0.00001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


1. Base on experiment 1.1 we can conclude that the a shallower network generates higher accuracy
2. The network is not trainable for L in {8, 16}.
    - L=8 we get very deep network (with minimal pooling in comparison to feature expansion).
    This results in model which has a lot of features that we are not able to train with our resources
    - L=16 we cannot build the model, let alone train it because we cant build the classifier since for each maxPool
    operation we need to divide each of the dimension of H, W by 2 and since the initial dimension are 32*32 we get a
    layer of num_input_feature=0 which is not possible.
"""

part3_q2 = r"""
**Your answer:**


In experiment 1.2 we can see that the value of K (number of filters) has a great impact on the tests accuracy
and as expected from the last experiment, the shallower network gave as the best results (L=2, K=258)
"""

part3_q3 = r"""
**Your answer:**


In this experiment the network fails to train.
<br/>
Our assumption is that if the filters job is to generate features then we expect that the next layer
will use these features to reduce the problem space, by using these features to make a decision.
<br/>
If this assumption is true, after layer will many filters we should get a layer with less filters.
<br/>
In this experiment we can see that the model is built in the opposite way
as we get to a deeper layer we get more filters.
<br/>
Also, adding more features should not prevent us from training the model,
just make the job harder since we have a lot of features to train and convolving the features can
generate a large set of values for the features (some features can have very large values while others have small ones)
"""


part3_q4 = r"""
**Your answer:**


1. we added to the feature extractor part batchNorm layer after each convolution layer, based on ```VGG model```
that is proven to be a good model for ```cifar10``` data set, though our classifier block is different
and our extractor block is a modification of the ```VGG model``` in order to maintain the assignments required behavior 
2. The normalization layers that we added help us keep all the values magnitude relatively close,
this helped us in the learning phase which results in much better accuracy.💋
"""
# ==============
