
import fedplat as fp
import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN_CIFAR10_FedAvg(fp.Model):
    def __init__(self, device, *args, **kwargs):
        super(CNN_CIFAR10_FedAvg, self).__init__(device)
        self.input_require_shape = [3, -1, -1]
        self.target_require_shape = []
    def generate_net(self, input_data_shape, target_class_num, *args, **kwargs):
        self.name = 'CNN_CIFAR10_FedAvg'
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1600, 384),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(192, target_class_num),
        )
        self.create_Loc_reshape_list()
    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        return self.decoder(x)
