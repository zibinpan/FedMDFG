
import fedplat as fp
import numpy as np
import copy
import torch
from torch.autograd import Variable
class AFL(fp.Algorithm):
    def __init__(self,
                 name='AFL',
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
                 lam=0.01,  
                 *args,
                 **kwargs):
        if params is not None:
            lam = params['lam']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' lam' + str(lam)
        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, update_client, write_log)
        self.lam = lam
        self.dynamic_lambdas = np.ones(self.client_num) * 1.0 / self.client_num
        self.result_model = copy.deepcopy(self.model)
    def run(self):
        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            self.send_train_order(self.epochs)
            m_locals, l_locals = self.send_require_client_model()
            g_locals = []  
            old_models = self.model.span_model_params_to_vec()
            for idx, client in enumerate(m_locals):
                g_locals.append((old_models - m_locals[idx].span_model_params_to_vec()) / self.lr)  
            g_locals = torch.stack(g_locals)  
            weights = torch.Tensor(self.dynamic_lambdas).float().to(self.device)
            d = weights @ g_locals
            for i, p in enumerate(self.model.parameters()):
                p.grad = d[self.model.Loc_reshape_list[i]]
            self.optimizer.step()
            self.dynamic_lambdas = [lmb_i+self.lam * float(loss_i) for lmb_i,loss_i in zip(self.dynamic_lambdas, l_locals)]
            self.dynamic_lambdas = self.project(self.dynamic_lambdas)
            self.result_model = (self.current_comm_round * self.result_model + self.model) * (1 / (self.current_comm_round + 1))
            self.current_training_num += self.epochs * batch_num
    def project(self, p):
        u = sorted(p, reverse=True)
        res = []
        rho = 0
        for i in range(len(p)):
            if (u[i] + (1.0/(i + 1)) * (1 - np.sum(np.asarray(u)[:i+1]))) > 0:
                rho = i + 1
        lamb = (1.0/(rho+1e-6)) * (1 - np.sum(np.asarray(u)[:rho]))
        for i in range(len(p)):
            res.append(max(p[i] + lamb, 0))
        return res
