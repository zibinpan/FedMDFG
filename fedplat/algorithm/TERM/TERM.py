
import fedplat as fp
import numpy as np
import copy
import torch
from torch.autograd import Variable
class TERM(fp.Algorithm):
    def __init__(self,
                 name='TERM',
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
                 t=1,  
                 *args,
                 **kwargs):
        if params is not None:
            t = params['t']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' t' + str(t)
        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, update_client, write_log)
        self.t = t
        self.estimates = 0
    def run(self):
        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            self.send_train_order(self.epochs)
            m_locals, l_locals = self.send_require_client_model()
            losses = np.array([float(l_local) for l_local in l_locals])
            old_model_params = self.model.span_model_params_to_vec()
            updates = [old_model_params - m_local.span_model_params_to_vec() for m_local in m_locals]
            max_l = np.max(losses)
            new_ = np.mean(np.exp(self.t * np.array((losses - max_l))))
            self.estimates = self.estimates * 0.5 + new_ * 0.5
            weights = np.exp(self.t * (losses - max_l)) / (self.estimates * self.client_num)
            c_sols = []
            for idx, u in enumerate(updates):
                updates[idx] = updates[idx] * weights[idx]
                c_sols.append(old_model_params - updates[idx])
            self.aggregate(c_sols)
            self.current_training_num += self.epochs * batch_num
    def aggregate(self, model_params_list):
        aggregate_weights = torch.Tensor([1 / self.client_num] * self.client_num).float().to(self.device)
        model_params_mat = torch.stack(model_params_list)
        model_params = aggregate_weights @ model_params_mat
        for i, p in enumerate(self.model.parameters()):
            p.data = model_params[self.model.Loc_reshape_list[i]]
