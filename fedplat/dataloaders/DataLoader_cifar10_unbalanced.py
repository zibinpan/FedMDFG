
import fedplat as fp
import os
import torch
import torchvision
import random
from torchvision import transforms as transforms
import numpy as np
import copy
class DataLoader_cifar10_unbalanced(fp.DataLoader):

    def __init__(self,
                 pool_size=5,
                 batch_size=50,
                 input_require_shape=None,
                 shuffle=True,
                 types=None,
                 recreate=False,
                 params=None,
                 *args,
                 **kwargs):
        if params is not None:
            pool_size = params['N']
            batch_size = params['B']
            types = params['types']  
        print('batch_size: ', batch_size)
        
        if types is None or types == '' or types == 'default_type':
            
            types = [[0], [1, 2], [3, 4, 5, 6], [7], [8, 9]]
        else:  
            types = types.split('_')
            for i in range(len(types)):
                types[i] = types[i].split('-')
                for j in range(len(types[i])):
                    types[i][j] = int(types[i][j])
        name = 'CIFAR10_pool_' + str(pool_size) + '_batchsize_' + str(batch_size) + '_types_' + str(types) + '_input_require_shape_' + str(input_require_shape)
        nickname = 'cifar10 unbalanced B' + str(batch_size) + ' N' + str(pool_size) + ' types' + str(types)
        super().__init__(name, nickname, pool_size, batch_size, input_require_shape)
        self.types = types
        
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
            trainset = torchvision.datasets.CIFAR10(root=fp.data_folder_path, train=True,
                                                    download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainset.data.shape[0],
                                                      shuffle=True, num_workers=1)
            testset = torchvision.datasets.CIFAR10(root=fp.data_folder_path, train=False,
                                                   download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=testset.data.shape[0],
                                                     shuffle=False, num_workers=1)
            global_training_data = torch.utils.data.DataLoader(copy.deepcopy(trainset), batch_size=self.batch_size,
                                                               shuffle=True, num_workers=1)
            global_test_data = torch.utils.data.DataLoader(copy.deepcopy(testset), batch_size=self.batch_size,
                                                           shuffle=False, num_workers=1)
            
            for i, (input_data, targets) in enumerate(trainloader):  
                train_input_data = input_data
                train_target_data = targets
            for i, (input_data, targets) in enumerate(testloader):  
                test_input_data = input_data
                test_target_data = targets
            
            self.cal_data_shape(train_input_data.shape)
            self.target_class_num = 10  
            
            self.global_training_data = []
            self.global_test_data = []
            for (input_data, targets) in global_training_data:
                self.global_training_data.append((input_data.reshape([-1] + self.input_data_shape), targets))
            for (input_data, targets) in global_test_data:
                self.global_test_data.append((input_data.reshape([-1] + self.input_data_shape), targets))
            self.total_training_number = len(trainset)
            self.total_test_number = len(testset)
            def create_data_pool(data_pool, input_data, target_data, key_name):
                
                data_indices_of_diff_types = []  
                for i in range(len(types)):
                    indices = []
                    for j in range(len(types[i])):
                        matched_idx = torch.where(target_data == types[i][j])[0]
                        indices.append(matched_idx)
                    data_indices_of_diff_types.append(torch.cat(indices))  
                for i in range(len(types)):
                    type_data_indices = data_indices_of_diff_types[i]
                    if shuffle:  
                        order = list(range(len(type_data_indices)))
                        random.shuffle(order)
                        type_data_indices = type_data_indices[order]
                    
                    batch_data_indices_list = fp.DataLoader.separate_list(type_data_indices, self.batch_size)
                    local_data = []
                    local_data_number = 0
                    for batch_data_indices in batch_data_indices_list:
                        
                        batch_input_data = input_data[batch_data_indices].reshape([-1] + self.input_data_shape).float()  
                        
                        batch_target_data = target_data[batch_data_indices]
                        local_data.append((batch_input_data, batch_target_data))
                        local_data_number += batch_input_data.shape[0]
                    
                    data_pool[i][key_name + '_data'] = local_data
                    data_pool[i][key_name + '_number'] = local_data_number
                    data_pool[i]['data_name'] = str(types[i])
            
            data_pool = [{} for _ in range(self.pool_size)]
            
            create_data_pool(data_pool, train_input_data, train_target_data, 'local_training')
            
            create_data_pool(data_pool, test_input_data, test_target_data, 'local_test')
            
            min_batch_num = min([len(pool_item['local_training_data']) for pool_item in data_pool])
            for pool_item in data_pool:
                pool_item['local_training_data'] = pool_item['local_training_data'][:min_batch_num]
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
