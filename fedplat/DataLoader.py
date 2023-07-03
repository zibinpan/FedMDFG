
import copy
import random
import torch
class DataLoader:
    def __init__(self,
                 name='DataLoader',
                 nickname='DataLoader',
                 pool_size=0,
                 batch_size=0,
                 input_require_shape=None,
                 *args,
                 **kwargs):
        self.name = name
        self.nickname = nickname
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.input_require_shape = input_require_shape
        self.input_data_shape = None
        self.target_class_num = None
        self.data_pool = None
        self.global_training_data = None
        self.global_test_data = None
    def allocate(self, client_list):
        raise RuntimeError('error in Algorithm: This function must be rewritten in the child class.')
    def reshape(self, data, require_shape):
        return data.reshape(require_shape)
    def transform_data(self, dataset):
        input_data = []
        for i, data_item in enumerate(dataset):
            input_data.append(data_item[0])
        input_data = torch.cat(input_data)
        target_data = copy.deepcopy(dataset.targets)
        return input_data, target_data
    def cal_data_shape(self, raw_input_data_shape):
        def cal(require_shape, raw_shape):
            if len(require_shape) == len(raw_shape) - 1:
                data_shape = list(raw_shape[1:])
            else:
                data_shape = []
                for i in range(1, len(raw_shape)):
                    if i < len(require_shape) + 1:
                        data_shape.append(raw_shape[i])
                    else:
                        data_shape[-1] *= raw_shape[i]
            return data_shape
        self.input_data_shape = cal(self.input_require_shape, raw_input_data_shape)
    @staticmethod
    def separate_list(input_list, n):
        def separate(input_list, n):
            for i in range(0, len(input_list), n):
                yield input_list[i: i + n]
        return list(separate(input_list, n))
    @staticmethod
    def random_choice(n1, n2):
        indices = list(range(n1))
        indices_copy = copy.deepcopy(indices)
        choose_indices = []
        choose_indices_reverse = []
        for i in range(n1):
            choose_indices_reverse.append([])
        for i in range(n2):
            if len(indices_copy) == 0:
                indices_copy = copy.deepcopy(indices)
            pick = indices_copy[random.randint(0, len(indices_copy) - 1)]
            choose_indices.append(pick)
            choose_indices_reverse[pick].append(i)
            indices_copy.remove(pick)
        return choose_indices, choose_indices_reverse
if __name__ == '__main__':
    input_list = [1,2,3,4,5,6,7,8,9]
    n = 10
    print(DataLoader.separate_list(input_list, n))
