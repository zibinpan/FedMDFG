
import fedplat as fp
import os
import torch
import torchvision
import random
from torchvision import transforms as transforms
import numpy as np
import copy
class DataLoader_cifar100(fp.DataLoader):

    def __init__(self,
                 split_num=100,
                 pick_num=1,
                 batch_size=100,
                 input_require_shape=None,
                 shuffle=True,
                 recreate=False,
                 params=None,
                 *args,
                 **kwargs):
        if params is not None:
            split_num = params['SN']
            pick_num = params['PN']
            batch_size = params['B']
        if split_num % pick_num != 0:
            raise RuntimeError('split_num must be divisible by the number of pick_num.')
        pool_size = split_num // pick_num
        name = 'CIFAR100_pool_' + str(pool_size) + 'split_' + str(split_num) + 'pick' + str(pick_num) + '_batchsize_' + str(batch_size) + '_sort_split_input_require_shape_' + str(input_require_shape)
        nickname = 'cifar100 B' + str(batch_size) + ' S'+ str(split_num) + ' P' + str(pick_num) + ' N' + str(pool_size)
        super().__init__(name, nickname, pool_size, batch_size, input_require_shape)
        
        file_path = fp.pool_folder_path + name + '.npy'
        if os.path.exists(file_path) and (recreate == False):
            data_loader = np.load(file_path, allow_pickle=True).item()  
            for attr in list(data_loader.__dict__.keys()):
                setattr(self, attr, data_loader.__dict__[attr])
            print('Successfully Read the Data Pool.')
        else:
            
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            trainset = torchvision.datasets.CIFAR100(root=fp.data_folder_path, train=True,
                                                     download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainset.data.shape[0],
                                                      shuffle=True, num_workers=1)
            testset = torchvision.datasets.CIFAR100(root=fp.data_folder_path, train=False,
                                                    download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=testset.data.shape[0],
                                                     shuffle=False, num_workers=1)
            global_training_data = torch.utils.data.DataLoader(copy.deepcopy(trainset),
                                                               batch_size=self.batch_size,
                                                               shuffle=True, num_workers=1)
            global_test_data = torch.utils.data.DataLoader(copy.deepcopy(testset),
                                                           batch_size=self.batch_size,
                                                           shuffle=False, num_workers=1)
            
            for i, (input_data, targets) in enumerate(trainloader):  
                train_input_data = input_data
                train_target_data = targets
            for i, (input_data, targets) in enumerate(testloader):  
                test_input_data = input_data
                test_target_data = targets
            
            self.cal_data_shape(train_input_data.shape)
            self.target_class_num = 100  
            
            self.global_training_data = []
            self.global_test_data = []
            for (input_data, targets) in global_training_data:
                self.global_training_data.append((input_data.reshape([-1] + self.input_data_shape), targets))
            for (input_data, targets) in global_test_data:
                self.global_test_data.append((input_data.reshape([-1] + self.input_data_shape), targets))
            self.total_training_number = len(trainset)
            self.total_test_number = len(testset)
            def create_data_pool(data_pool, input_data, target_data, key_name):
                
                order = torch.argsort(target_data)
                input_data = input_data[order, :]
                target_data = target_data[order]
                
                count = 0
                amount = input_data.shape[0] // split_num  
                indices = list(range(input_data.shape[0]))
                split_data_indices_list = []  
                for split_idx in range(split_num):
                    start_idx = count
                    end_idx = count + amount
                    if end_idx > input_data.shape[0] - 1:
                        end_idx = input_data.shape[0] - 1
                    split_data_indices = indices[start_idx: end_idx]  
                    split_data_indices_list.append(split_data_indices)
                    count += amount
                for pool_idx in range(pool_size):
                    data_indices = []
                    
                    for i in range(pick_num):
                        pick_data_indices = split_data_indices_list[random.randint(0, len(split_data_indices_list) - 1)]
                        data_indices += pick_data_indices
                        split_data_indices_list.remove(pick_data_indices)  
                    random.shuffle(data_indices)  
                    local_data_number = len(data_indices)
                    
                    batch_data_indices_list = fp.DataLoader.separate_list(data_indices, self.batch_size)
                    local_data = []
                    for batch_data_indices in batch_data_indices_list:
                        
                        batch_input_data = input_data[batch_data_indices].reshape([-1] + self.input_data_shape).float()  
                        
                        batch_target_data = target_data[batch_data_indices]
                        local_data.append((batch_input_data, batch_target_data))
                    
                    data_pool[pool_idx][key_name + '_data'] = local_data
                    data_pool[pool_idx][key_name + '_number'] = local_data_number
                    data_pool[pool_idx]['data_name'] = str(pool_idx)
            
            data_pool = [{} for _ in range(self.pool_size)]
            
            create_data_pool(data_pool, train_input_data, train_target_data, 'local_training')
            
            create_data_pool(data_pool, test_input_data, test_target_data, 'local_test')
            self.data_pool = data_pool
            
            np.save(file_path, self)
    def allocate(self, client_list):

        choose_data_pool_item_indices = np.random.choice(list(range(self.pool_size)), len(client_list), replace=False)
        for idx, client in enumerate(client_list):
            data_pool_item = self.data_pool[choose_data_pool_item_indices[idx]]
            client.update_data(choose_data_pool_item_indices[idx],
                               data_pool_item['local_training_data'],
                               data_pool_item['local_training_number'],
                               data_pool_item['local_test_data'],
                               data_pool_item['local_test_number'])
