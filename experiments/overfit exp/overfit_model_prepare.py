
import fedplat as fp
import torch
import os
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np


seed = 0

lr = 0.1  # {0.01, 0.05, 0.1}

def outFunc(alg):
    print(alg.save_name)
    print('round {}'.format(alg.current_comm_round), 'training_num {}'.format(alg.current_training_num))
    print('learning rate: ', alg.lr)
    print('Global Test loss: ', format(alg.comm_log['global_test_loss'][-1], '.6f'),
          'Global Test Accuracy: ', format(alg.comm_log['global_test_accuracy'][-1], '.6f'))
    loss_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        loss_list.append(metric_history['training_loss'][-1])
    print(f'Training loss: ave: {format(np.mean(loss_list), ".6f")}, std: {format(np.std(loss_list), ".6f")}, min: {format(np.min(loss_list), ".6f")}, max: {format(np.max(loss_list), ".6f")}')
    value_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        value_list.append(metric_history['test_accuracy'][-1])
    print(f'Test Acc: ave: {format(np.mean(value_list), ".6f")}, std: {format(np.std(value_list), ".6f")}, min: {format(np.min(value_list), ".6f")}, max: {format(np.max(value_list), ".6f")}')
    print()
    if np.min(loss_list) <= 1e-6:
        torch.save(alg.model, str(seed) + 'unfair_model.pth')
        exit(0)


fp.setup_seed(seed=seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = fp.CNN_CIFAR10_FedAvg(device)

data_loader = fp.DataLoader_cifar10(split_num=10, pick_num=1, batch_size=200, input_require_shape=model.input_require_shape, recreate=False)

model.generate_net(data_loader.input_data_shape, data_loader.target_class_num)

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0)

train_setting = {'criterion': torch.nn.CrossEntropyLoss(), 'optimizer': optimizer, 'lr_decay': 0.999}

algorithm = fp.FedAvg(data_loader=data_loader,
                      model=model,
                      device=device,
                      train_setting=train_setting,
                      client_num=1,  
                      metric_list=[fp.Correct()],  
                      max_comm_round=3000,  
                      max_training_num=None,  
                      epochs=1,
                      outFunc=outFunc,
                      update_client=False)
algorithm.save_folder = 'resutls/'
if not os.path.exists(algorithm.save_folder):
    os.makedirs(algorithm.save_folder)
algorithm.save_name = algorithm.name + ' seed1 lr0.1 decay0.999'

algorithm.run()
