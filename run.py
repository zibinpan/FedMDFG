# -*- coding: utf-8 -*-
import fedplat as fp
import os


if __name__ == '__main__':
    params = fp.read_params()
    data_loader, algorithm = fp.initialize(params)
    algorithm.save_folder = data_loader.nickname + '/C' + str(params['C']) + '/' + params['model'] + '/' + params['algorithm'] + '/'
    if not os.path.exists(algorithm.save_folder):
        os.makedirs(algorithm.save_folder)
    algorithm.save_name = 'seed' + str(params['seed']) + ' N' + str(data_loader.pool_size) + ' C' + str(params['C']) + ' ' + algorithm.save_name
    algorithm.run()
