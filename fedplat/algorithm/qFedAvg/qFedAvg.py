
import fedplat as fp
import torch
import numpy as np
class qFedAvg(fp.Algorithm):
    def __init__(self,
                 name='qFedAvg',
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
                 q=0.1,  
                 *args,
                 **kwargs):
        if params is not None:
            q = params['q']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' q' + str(q)
        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, update_client, write_log)
        self.q = q
        self.lr = self.train_setting['optimizer'].defaults['lr']
    def run(self):
        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            self.send_train_order(self.epochs)
            m_locals, l_locals = self.send_require_client_model()
            g_locals = []  
            for idx, client in enumerate(m_locals):
                grad = (self.model - m_locals[idx]) * (1 / self.lr)
                g_locals.append(grad)
            Deltas = [gi * (li + 1e-10)**self.q for gi, li in zip(g_locals, l_locals)]
            hs = [self.q * (li + 1e-10)**(self.q - 1) * gi.L2_norm_square() +
                  1.0 / self.lr * (li + 1e-10)**self.q for gi, li in zip(g_locals, l_locals)]
            self.model = self.aggregate(Deltas, hs)
            self.current_training_num += self.epochs * batch_num
    def aggregate(self, Deltas, hs):
        denominator = float(np.sum([v.cpu() for v in hs]))
        scaled_deltas = [delta * (1.0 / denominator) for delta in Deltas]
        updates = fp.Model.model_sum(scaled_deltas)
        new_model = self.model - updates
        return new_model
