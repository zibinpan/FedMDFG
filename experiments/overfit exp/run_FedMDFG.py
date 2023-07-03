
import fedplat as fp
import torch
import os
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np


seed = 0

lr = 0.1  # {0.01, 0.05, 0.1}

fp.setup_seed(seed=seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = fp.CNN_CIFAR10_FedAvg(device)

data_loader = fp.DataLoader_cifar10(split_num=10, pick_num=1, batch_size=200, input_require_shape=model.input_require_shape, recreate=False)

model.generate_net(data_loader.input_data_shape, data_loader.target_class_num)

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0)

train_setting = {'criterion': torch.nn.CrossEntropyLoss(), 'optimizer': optimizer, 'lr_decay': 0.999}

algorithm = fp.FedMGDP_Armijo(data_loader=data_loader,
                      model=model,
                      device=device,
                      train_setting=train_setting,
                      client_num=10,  
                      metric_list=[fp.Correct()],  
                      max_comm_round=3000,  
                      max_training_num=None,  
                      epochs=1,
                      outFunc=fp.outFunc,
                      alpha=5.625,
                      s=5,
                      update_client=False)
algorithm.save_folder = 'resutls/'
if not os.path.exists(algorithm.save_folder):
    os.makedirs(algorithm.save_folder)
algorithm.save_name = algorithm.name + ' seed1 lr0.1 decay0.999'

algorithm.model = torch.load('unfair_model.pth')

algorithm.run()
