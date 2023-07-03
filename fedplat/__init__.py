import os
from fedplat.Algorithm import Algorithm
from fedplat.Client import Client
from fedplat.DataLoader import DataLoader
from fedplat.Model import Model
from fedplat.Metric import Metric
from fedplat.seed import setup_seed
from fedplat.metric.Correct import Correct
from fedplat.metric.Precision import Precision
from fedplat.metric.Recall import Recall
from fedplat.model.CNN_of_cifar10_tutorial import CNN_of_cifar10_tutorial
from fedplat.model.CNN_OriginalFedAvg import CNN_OriginalFedAvg
from fedplat.model.CNN_CIFAR10_FedAvg import CNN_CIFAR10_FedAvg
from fedplat.model.MLP import MLP
import fedplat.algorithm
from fedplat.algorithm.FedAvg.FedAvg import FedAvg
from fedplat.algorithm.qFedAvg.qFedAvg import qFedAvg
from fedplat.algorithm.AFL.AFL import AFL
from fedplat.algorithm.FedFV.FedFV import FedFV
from fedplat.algorithm.Ditto.Ditto import Ditto
from fedplat.algorithm.TERM.TERM import TERM
from fedplat.algorithm.FedMDFG.FedMDFG import FedMDFG
from fedplat.dataloaders.DataLoader_cifar10 import DataLoader_cifar10
from fedplat.dataloaders.DataLoader_cifar10_unbalanced import DataLoader_cifar10_unbalanced
from fedplat.dataloaders.DataLoader_mnist import DataLoader_mnist
from fedplat.dataloaders.DataLoader_mnist_unbalanced import DataLoader_mnist_unbalanced
from fedplat.dataloaders.DataLoader_fashion import DataLoader_fashion
from fedplat.dataloaders.DataLoader_fashion_unbalanced import DataLoader_fashion_unbalanced
from fedplat.dataloaders.DataLoader_cifar100 import DataLoader_cifar100
data_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
pool_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/pool/'
if not os.path.exists(pool_folder_path):
    os.makedirs(pool_folder_path)
from fedplat.main import initialize, read_params, outFunc
