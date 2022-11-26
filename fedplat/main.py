
import fedplat as fp
import numpy as np
import argparse
import torch
import sys
torch.multiprocessing.set_sharing_strategy('file_system')
def outFunc(alg):
    loss_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        loss_list.append(metric_history['training_loss'][-1])
    value_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        value_list.append(metric_history['test_accuracy'][-1])
    value_list = np.array(value_list)
    p = np.ones(len(value_list))
    stream_log = ""
    stream_log += alg.save_name + '\n'
    stream_log += 'round {}'.format(alg.current_comm_round) + ' training_num {}'.format(alg.current_training_num) + '\n'
    stream_log += 'Global Test loss: ' + format(alg.comm_log['global_test_loss'][-1], '.6f') + ' Global Test Accuracy: ' + format(alg.comm_log['global_test_accuracy'][-1], '.6f') + '\n'
    stream_log += f'Training loss: ave: {format(np.mean(loss_list), ".6f")}, std: {format(np.std(loss_list), ".6f")}, min: {format(np.min(loss_list), ".6f")}, max: {format(np.max(loss_list), ".6f")}' + '\n'
    stream_log += f'Test Acc: ave: {format(np.mean(value_list), ".6f")}, std: {format(np.std(value_list), ".6f")}, angle: {format(np.arccos(value_list @ p / (np.linalg.norm(value_list) * np.linalg.norm(p))), ".6f")}, min: {format(np.min(value_list), ".6f")}, max: {format(np.max(value_list), ".6f")}' + '\n'
    stream_log += '\n'
    alg.stream_log += stream_log
    print(stream_log)
def read_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='seed', type=int, default=1)
    parser.add_argument('--device', help='device: -1, 0, 1, or ...', type=int, default=0)
    parser.add_argument('--model', help='model name;', type=str, default='CNN_CIFAR10_FedAvg')
    parser.add_argument('--algorithm', help='algorithm name;', type=str, default='FedAvg')
    parser.add_argument('--dataloader', help='dataloader name;', type=str, default='DataLoader_cifar10_non_iid')
    parser.add_argument('--SN', help='split num', type=int, default=200)
    parser.add_argument('--PN', help='pick num', type=int, default=2)
    parser.add_argument('--B', help='batch size', type=int, default=50)
    parser.add_argument('--types', help='dataloader label types;', type=str, default='default_type')
    parser.add_argument('--N', help='client num', type=int, default=100)
    parser.add_argument('--C', help='select client proportion', type=float, default=1.0)
    parser.add_argument('--R', help='communication round', type=int, default=3000)
    parser.add_argument('--E', help='local epochs', type=int, default=1)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.1)  
    parser.add_argument('--decay', help='learning rate decay', type=float, default=0.999)  
    parser.add_argument('--alpha', help='alpha of FedFV', type=float, default=0.1)
    parser.add_argument('--theta', help='theta of FedMDFG', type=float, default=11.25)
    parser.add_argument('--s', help='line search parameter of FedMDFG', type=int, default=5)  
    parser.add_argument('--tau', help='parameter tau in FedFV', type=int, default=1)  
    parser.add_argument('--lam', help='parameter tau in FedFV', type=float, default=0.1)
    parser.add_argument('--epsilon', help='parameter epsilon in FedMGDA+', type=float, default=0.1)
    parser.add_argument('--q', help='parameter q in qFedAvg', type=float, default=0.1)
    parser.add_argument('--t', help='parameter t in TERM', type=int, default=1)
    parser.add_argument('--dishonest_num', help='dishonest number', type=int, default=0)
    try:
        parsed = vars(parser.parse_args())
        return parsed
    except IOError as msg:
        parser.error(str(msg))
def initialize(params):
    fp.setup_seed(seed=params['seed'])
    device = torch.device('cuda:' + str(params['device']) if torch.cuda.is_available() and params['device'] != -1 else "cpu")
    Model = getattr(sys.modules['fedplat'], params['model'])
    model = Model(device)
    Dataloader = getattr(sys.modules['fedplat'], params['dataloader'])
    data_loader = Dataloader(params=params, input_require_shape=model.input_require_shape)
    model.generate_net(data_loader.input_data_shape, data_loader.target_class_num)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'])
    train_setting = {'criterion': torch.nn.CrossEntropyLoss(), 'optimizer': optimizer, 'lr_decay': params['decay']}
    Algorithm = getattr(sys.modules['fedplat'], params['algorithm'])
    algorithm = Algorithm(data_loader=data_loader,
                          model=model,
                          device=device,
                          train_setting=train_setting,
                          client_num=int(data_loader.pool_size * params['C']),  
                          metric_list=[fp.Correct()],  
                          max_comm_round=params['R'],  
                          max_training_num=None,  
                          epochs=params['E'],
                          outFunc=outFunc,
                          update_client=True,
                          params=params,
                          write_log=True)
    return data_loader, algorithm
if __name__ == '__main__':
    params = read_params()
    data_loader, algorithm = initialize(params)
    algorithm.run()
