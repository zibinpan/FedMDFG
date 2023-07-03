
import fedplat as fp
import torch
import numpy as np
from fedplat.algorithm.common.utils import get_d_mgdaplus_d
class FedMGDA_plus(fp.Algorithm):
    def __init__(self,
                 name='FedMGDA+',
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
                 update_client=False,
                 write_log=True,
                 params=None,
                 epsilon=0.1,  
                 *args,
                 **kwargs):
        if params is not None:
            epsilon = params['epsilon']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' epsilon' + str(epsilon)
        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, update_client, write_log)
        self.epsilon = epsilon
        self.comm_log['d_optimality_history'] = []  
        self.comm_log['d_descent_history'] = []  
    def run(self):
        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            self.model.train()  
            self.send_train_order(self.epochs)
            m_locals, l_locals = self.send_require_client_model()
            g_locals = []  
            old_models = self.model.span_model_params_to_vec()
            for idx, client in enumerate(m_locals):
                grad = old_models - m_locals[idx].span_model_params_to_vec()  
                g_locals.append(grad)
            g_locals = torch.stack(g_locals)
            g_locals /= torch.norm(g_locals, dim=1).reshape(-1, 1)
            training_nums = self.send_require_attr('local_training_number')
            lambda0 = np.array(training_nums) / sum(training_nums)
            d, d_optimal_flag, d_descent_flag = get_d_mgdaplus_d(g_locals, self.device, self.epsilon, lambda0)
            for i, p in enumerate(self.model.parameters()):
                p.grad = d[self.model.Loc_reshape_list[i]]
            self.optimizer.step()
            self.current_training_num += self.epochs * batch_num
            self.comm_log['d_optimality_history'].append(d_optimal_flag)
            self.comm_log['d_descent_history'].append(d_descent_flag)
