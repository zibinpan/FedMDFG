## For more details, follow the `readme`.
# FedAvg
python run.py --seed 1 --device 1 --model CNN_CIFAR10_FedAvg --algorithm FedAvg --dataloader DataLoader_cifar10 --SN 200 --PN 2 --B 50 --C 0.1 --R 2000 --E 1 --lr 0.1 --decay 0.999
python run.py --seed 2 --device 1 --model CNN_CIFAR10_FedAvg --algorithm FedAvg --dataloader DataLoader_cifar10 --SN 200 --PN 2 --B 50 --C 0.1 --R 2000 --E 1 --lr 0.1 --decay 0.999
python run.py --seed 3 --device 1 --model CNN_CIFAR10_FedAvg --algorithm FedAvg --dataloader DataLoader_cifar10 --SN 200 --PN 2 --B 50 --C 0.1 --R 2000 --E 1 --lr 0.1 --decay 0.999
python run.py --seed 4 --device 1 --model CNN_CIFAR10_FedAvg --algorithm FedAvg --dataloader DataLoader_cifar10 --SN 200 --PN 2 --B 50 --C 0.1 --R 2000 --E 1 --lr 0.1 --decay 0.999
python run.py --seed 5 --device 1 --model CNN_CIFAR10_FedAvg --algorithm FedAvg --dataloader DataLoader_cifar10 --SN 200 --PN 2 --B 50 --C 0.1 --R 2000 --E 1 --lr 0.1 --decay 0.999

# FedMDFG
python run.py --seed 1 --device 1 --model CNN_CIFAR10_FedAvg --algorithm FedMDFG --dataloader DataLoader_cifar10 --SN 200 --PN 2 --B 50 --C 0.1 --R 2000 --E 1 --lr 0.1 --decay 0.999 --theta 11.25 --s 5
