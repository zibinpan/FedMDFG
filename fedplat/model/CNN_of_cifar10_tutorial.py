
import fedplat as fp
import torch.nn as nn
import torch.nn.functional as F
class CNN_of_cifar10_tutorial(fp.Model):
    def __init__(self, device, *args, **kwargs):
        super(CNN_of_cifar10_tutorial, self).__init__(device)
        self.input_require_shape = [3, -1, -1]
        self.target_require_shape = []
    def generate_net(self, input_data_shape, target_class_num, *args, **kwargs):
        self.name = 'CNN_of_cifar10_tutorial'
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, target_class_num)
        self.create_Loc_reshape_list()
    def forward(self, x):
        x = self.pool(nn.ReLU(self.conv1(x)))
        x = self.pool(nn.ReLU(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x
