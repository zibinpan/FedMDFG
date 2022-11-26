# -*- coding: utf-8 -*-
import fedplat as fp
import numpy as np
import os


if __name__ == '__main__':
    params = fp.read_params()
    data_loader, algorithm = fp.initialize(params)
    algorithm.save_folder = data_loader.nickname + '/C' + str(params['C']) + '/' + params['model'] + '/' + params['algorithm'] + '/'
    if not os.path.exists(algorithm.save_folder):
        os.makedirs(algorithm.save_folder)
    algorithm.save_name = 'seed' + str(params['seed']) + ' N' + str(data_loader.pool_size) + ' C' + str(params['C']) + ' ' + algorithm.save_name
    dishonest_num = params['dishonest_num']
    if dishonest_num >= data_loader.pool_size:
        raise RuntimeError('Error parameter dishonest_num')
    if dishonest_num > 0:
        dishonest_indices = np.random.choice(list(range(data_loader.pool_size)), dishonest_num ,replace=False).tolist()
        dishonest_list = [None] * data_loader.pool_size
        for idx in dishonest_indices:
            dishonest_list[idx] = {'grad norm': None, 'inverse grad': None, 'zero grad': None, 'random grad': None, 'random grad 10': None, 'gaussian': True}
        algorithm.dishonest_list = dishonest_list
        for client in algorithm.client_list:
            client.dishonest = algorithm.dishonest_list[client.id]
    algorithm.run()
