
import fedplat as fp
import torch
import numpy as np
import copy
import math
from torch.autograd import Variable


class FedFV(fp.Algorithm):
    def __init__(self,
                 name='FedFV',
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
                 alpha=0.1,  
                 tau=1,  
                 *args,
                 **kwargs):
        if params is not None:
            alpha = params['alpha']
            tau = params['tau']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' alpha' + str(alpha) + ' tau' + str(tau)
        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, update_client, write_log)
        self.alpha = alpha
        self.tau = tau
        self.lr = self.train_setting['optimizer'].defaults['lr']
        self.client_last_sample_round = [-1 for i in range(self.data_loader.pool_size)]
        self.client_grads_history = [0 for i in range(self.data_loader.pool_size)]
        
    def run(self):
        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            self.send_train_order(self.epochs)
            m_locals, l_locals = self.send_require_client_model()
            g_locals = []  
            old_models = self.model.span_model_params_to_vec()
            for idx, client in enumerate(m_locals):
                g_locals.append(old_models - m_locals[idx].span_model_params_to_vec())  
            for cid, gi in zip(self.send_require_attr('id'), g_locals):
                self.client_grads_history[cid] = gi
                self.client_last_sample_round[cid] = self.current_comm_round
            order_grads = copy.deepcopy(g_locals)
            order = [_ for _ in range(len(order_grads))]
            tmp = sorted(list(zip(l_locals, order)), key=lambda x: x[0])
            order = [x[1] for x in tmp]
            keep_original = []
            if self.alpha > 0:
                keep_original = order[math.ceil((len(order) - 1) * (1 - self.alpha)):]
            g_locals_L2_norm_square_list = []
            for g_local in g_locals:
                g_locals_L2_norm_square_list.append(torch.norm(g_local)**2)
            for i in range(len(order_grads)):
                if i in keep_original:
                    continue
                for j in order:
                    if j == i:
                        continue
                    else:
                        dot = g_locals[j] @ order_grads[i]
                        if dot < 0:
                            order_grads[i] = order_grads[i] - dot / g_locals_L2_norm_square_list[j] * g_locals[j]
            weights = torch.Tensor([1 / len(order_grads)] * len(order_grads)).float().to(self.device)
            gt = weights @ torch.stack([order_grads[i] for i in range(len(order_grads))])
            if self.current_comm_round >= self.tau:
                for k in range(self.tau-1, -1, -1):
                    gcs = [self.client_grads_history[cid] for cid in range(self.data_loader.pool_size) if self.client_last_sample_round[cid] == self.current_comm_round - k and gt @ self.client_grads_history[cid] < 0]
                    if gcs:
                        gcs = torch.vstack(gcs)
                        g_con = torch.sum(gcs, dim=0)
                        dot = gt @ g_con
                        if dot < 0:
                            gt = gt - dot / (torch.norm(g_con)**2) * g_con
            gnorm = torch.norm(weights @ torch.stack([g_locals[i] for i in range(len(g_locals))]))
            gt = gt / torch.norm(gt) * gnorm
            for i, p in enumerate(self.model.parameters()):
                p.data -= gt[self.model.Loc_reshape_list[i]]
            self.current_training_num += self.epochs * batch_num
