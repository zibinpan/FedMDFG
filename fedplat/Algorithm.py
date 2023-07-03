
import fedplat as fp
import numpy as np
import torch
import copy
import json
class Algorithm:
    def __init__(self,
                 name='Algorithm',
                 data_loader=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 metric_list=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 update_client=True,
                 write_log=True,
                 dishonest_list=None,
                 *args,
                 **kwargs):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if dishonest_list is None:
            dishonest_list = [None] * data_loader.pool_size
        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [fp.Client(i, copy.deepcopy(model), device, train_setting, metric_list) for i in range(client_num)]  
            data_loader.allocate(client_list)  
            for client in client_list:
                client.dishonest = dishonest_list[client.id]
        elif client_num is None and client_list is None:
            raise RuntimeError('Both of client_num and client_list cannot be None or not None.')
        self.test_client_list = [fp.Client(i, copy.deepcopy(model), device, train_setting, metric_list) for i in range(data_loader.pool_size)]  
        data_loader.allocate(self.test_client_list)  
        if len(self.test_client_list) == 0:
            raise RuntimeError('All clients are dishonest.')
        if data_loader.pool_size == client_num:  
            update_client = False
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay'])
        if max_comm_round is None:
            max_comm_round = 10**10
        if max_training_num is None:
            max_training_num = 10**10
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.model = model
        self.train_setting = train_setting
        self.client_num = client_num
        self.client_list = client_list
        self.max_comm_round = max_comm_round
        self.max_training_num = max_training_num
        self.epochs = epochs
        self.save_name = save_name
        self.outFunc = outFunc
        self.update_client = update_client
        self.comm_trace = None
        self.current_comm_round = 0
        self.current_training_num = 0
        self.model.to(self.device)
        self.metric_list = metric_list
        self.write_log = write_log
        self.dishonest_list = dishonest_list
        self.stream_log = ""
        self.save_folder=''
        self.comm_log = {'average_training_loss': [],  
                         'std_training_loss': [],  
                         'global_test_loss': [],  
                         'client_metric_history': [],  
                         'training_num': []}  
        for metric in metric_list:
            if metric.name == 'correct':
                self.comm_log['global_test_accuracy'] = []
        self.current_comm_round = 0
        self.current_training_num = 0
        self.lr = self.train_setting['optimizer'].defaults['lr']
        self.initial_lr = self.lr
        self.optimizer = train_setting['optimizer'].__class__(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.optimizer.defaults = train_setting['optimizer'].defaults
        self.result_model = None
    def run(self):
        raise RuntimeError('error in Algorithm: This function must be rewritten in the child class.')
    @staticmethod
    def update_learning_rate(optimizer, lr):
        optimizer.defaults['lr'] = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
    def adjust_learning_rate(self):
        self.lr = self.initial_lr * self.train_setting['lr_decay']**self.current_comm_round
        self.optimizer = self.update_learning_rate(self.optimizer, self.lr)  
        self.send_update_learning_rate_order()
    def terminated(self, update_count=False):
        self.adjust_learning_rate()
        self.send_sync_model(update_count)
        self.comm_log['training_num'].append(self.current_training_num)
        self.test(self.result_model)
        if callable(self.outFunc):
            self.outFunc(self)
        if self.current_comm_round >= self.max_comm_round or self.current_training_num >= self.max_training_num:
            return True
        else:
            if self.update_client:
                self.data_loader.allocate(self.client_list)
                ids = [self.client_list[i].id for i in range(self.client_num)]
                print(ids)
                for client in self.client_list:
                    client.dishonest = self.dishonest_list[client.id]
            return False
    def send_sync_model(self, update_count=True, model=None, client_list=None):
        if model is None:
            model = self.model
        if client_list is None:
            client_list = self.client_list
        for idx, client in enumerate(client_list):
            msg = {'command': 'sync', 'w_global': copy.deepcopy(model.state_dict())}
            client.get_message(msg)
        if update_count:
            self.current_comm_round += 1  
    def send_update_learning_rate_order(self):
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'update_learning_rate', 'current_comm_round': self.current_comm_round}
            client.get_message(msg)
    def send_cal_loss_order(self, batch_idx):
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'cal_loss', 'batch_idx': batch_idx}
            client.get_message(msg)
    def send_cal_all_batches_loss_order(self):
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'cal_all_batches_loss'}
            client.get_message(msg)
    def send_cal_all_batches_gradient_loss_order(self):
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'cal_all_batches_gradient_loss'}
            client.get_message(msg)
    def send_evaluate_order(self, batch_idx, mode='train'):
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'evaluate', 'mode': mode, 'batch_idx': batch_idx}
            client.get_message(msg)
    def send_train_order(self, epochs):
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'train', 'epochs': epochs, 'lr': self.lr}
            client.get_message(msg)
    def send_test_order(self):
        for idx, client in enumerate(self.test_client_list):
            msg = {'command': 'test'}
            client.get_message(msg)
    def send_require_cal_loss_result(self):
        l_locals = []  
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'require_cal_loss_result'}
            msg = client.get_message(msg)
            l_locals.append(msg['l_local'])
        l_locals = torch.stack(l_locals)
        return l_locals
    def send_require_cal_all_batches_loss_result(self):
        l_locals = []  
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'require_cal_all_batches_loss_result'}
            msg = client.get_message(msg)
            l_locals.append(msg['l_local'])
        l_locals = torch.stack(l_locals)
        return l_locals
    def send_require_all_batches_gradient_loss_result(self):
        g_locals = []  
        l_locals = []  
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'require_all_batches_gradient_loss_result'}
            msg = client.get_message(msg)
            g_locals.append(msg['g_local'])
            l_locals.append(msg['l_local'])
        g_locals = torch.stack([g_locals[i] for i in range(len(g_locals))])
        l_locals = torch.stack(l_locals)
        return g_locals, l_locals
    def send_require_evaluate_result(self):
        g_locals = []  
        l_locals = []  
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'require_evaluate_result'}
            msg = client.get_message(msg)
            g_locals.append(msg['g_local'])
            l_locals.append(msg['l_local'])
        g_locals = torch.stack([g_locals[i] for i in range(len(g_locals))])
        l_locals = torch.stack(l_locals)
        return g_locals, l_locals
    def send_require_client_model(self):
        m_locals = []
        l_locals = []
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'require_client_model', 'requires_grad': 'False'}
            msg = client.get_message(msg)
            m_locals.append(msg['m_local'])
            l_locals.append(msg['l_local'])
        return m_locals, l_locals
    def send_require_training_result(self):
        w_locals = []
        l_locals = []
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'require_training_result'}
            msg = client.get_message(msg)
            w_locals.append(msg['w_local'])
            l_locals.append(msg['l_local'])
        return w_locals, l_locals
    def send_require_attr(self, attr='local_training_number'):
        attrs = []
        for idx, client in enumerate(self.client_list):
            msg = {'command': 'require_attribute_value', 'attr': attr}
            msg = client.get_message(msg)
            attrs.append(msg['attr'])
        return attrs
    def test(self, test_model=None):
        if test_model is None:
            test_model = self.model
        self.send_sync_model(update_count=False, model=test_model, client_list=self.test_client_list)
        self.send_test_order()
        self.comm_log['client_metric_history'] = []
        for idx, client in enumerate(self.test_client_list):
            msg = {'command': 'require_test_result'}
            msg = client.get_message(msg)  
            self.comm_log['client_metric_history'].append(msg['metric_history'])
        total_test_number = sum([metric_history['local_test_number'] for metric_history in self.comm_log['client_metric_history']])  
        if 'global_test_accuracy' in self.comm_log:
            self.comm_log['global_test_accuracy'].append(100 * sum([metric_history['correct'][-1] for metric_history in self.comm_log['client_metric_history']]) / total_test_number)
        self.comm_log['global_test_loss'].append(sum([metric_history['test_loss'][-1] * metric_history['local_test_number'] for metric_history in self.comm_log['client_metric_history']]) / total_test_number)
        training_loss_list = []
        for i, metric_history in enumerate(self.comm_log['client_metric_history']):
            training_loss_list.append(metric_history['training_loss'][-1])
        self.comm_log['average_training_loss'] = np.mean(training_loss_list)
        self.comm_log['std_training_loss'] = np.std(training_loss_list)
        if self.write_log:
            self.save_log()

    def save_log(self):
        save_dict = {'algorithm name': self.name}
        save_dict['client num'] = self.client_num
        save_dict['communication round'] = self.current_comm_round
        save_dict['epochs'] = self.epochs
        save_dict['communication log'] = self.comm_log
        save_dict['info'] = 'data loader name_' + self.data_loader.name + '_model name_' + self.model.name + '_train setting_' + str(self.train_setting) + '_client num_' + str(self.client_num) + '_max comm round_' + str(self.max_comm_round) + '_epochs_' + str(self.epochs)
        file_name = self.save_folder + self.save_name + '.json'
        fileObject = open(file_name, 'w')
        fileObject.write(json.dumps(save_dict))
        fileObject.close()
        file_name = self.save_folder + 'log_' + self.save_name + '.log'
        fileObject = open(file_name, 'w')
        fileObject.write(self.stream_log)
        fileObject.close()
