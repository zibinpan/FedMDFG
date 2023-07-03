
import fedplat as fp
import numpy as np
import copy
import torch
from torch.autograd import Variable
class Ditto(fp.Algorithm):
    def __init__(self,
                 name='Ditto',
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
                 lam=0.1,  
                 *args,
                 **kwargs):
        if params is not None:
            lam = params['lam']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' lam' + str(lam)
        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, update_client, write_log)
        self.lam = lam
        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [Client(i, copy.deepcopy(model), device, train_setting, metric_list, lam) for i in range(client_num)]  
            data_loader.allocate(client_list)  
        elif client_num is None and client_list is None:
            raise RuntimeError('Both of client_num and client_list cannot be None or not None.')
        self.client_num = client_num
        self.client_list = client_list
    def run(self):
        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            self.send_train_order(self.epochs)
            m_locals, l_locals = self.send_require_client_model()
            self.aggregate(m_locals)
            self.current_training_num += self.epochs * batch_num
    def aggregate(self, m_locals):
        for m_local in m_locals:
            m_local_params = m_local.state_dict()
            model_params = self.model.state_dict()
            for layer in model_params.keys():
                model_params[layer] += (m_local_params[layer] - model_params[layer]) / self.client_num

class Client(fp.Client):
    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 lam=0,
                 *args,
                 **kwargs):
        super().__init__(id, model, device, train_setting, metric_list, *args, **kwargs)
        self.lam = lam
    def train(self, epochs):
        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')
        self.model.train()
        global_model_params = copy.deepcopy(self.model).span_model_params_to_vec()
        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                out = self.model(batch_x)
                loss = self.criterion(out, batch_y)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                grad_vec = self.model.span_model_grad_to_vec()
                eff_grad = grad_vec + self.lam * (self.model.span_model_params_to_vec() - global_model_params)
                for i, p in enumerate(self.model.parameters()):
                    p.data -= self.lr * eff_grad[self.model.Loc_reshape_list[i]]
        self.model_loss = Variable(loss, requires_grad=False)
