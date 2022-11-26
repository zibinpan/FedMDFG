# Code Appendix
Code for federated learning (FL).

## Requirements to run the code:

Python 3.6

Numpy

pytorch

cvxopt

## Basic usage
Copy one line of commands in the same folder of `./fedplat` and run (one example shown as follows).

```
python run.py --seed 1 --device 1 --model CNN_CIFAR10_FedAvg --algorithm FedAvg --dataloader DataLoader_cifar10 --SN 200 --PN 2 --B 50 --C 0.1 --R 2000 --E 1 --lr 0.1 --decay 0.999
```

All parameters can be seen in `./fedplat/main.py`.

By setting different parameters and run the command, you can replicate results of all experiments.

Enjoy yourself!

Paper Hash code:
FCEA460F92DAB6C51CE08FDFACF35219

Appendix Hash code:
F5ED0943FADF4DEE076088E14E4BF747
