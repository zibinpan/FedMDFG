
import fedplat as fp
import numpy as np
import torch
class FedAvg(fp.Algorithm):
    def __init__(self,
                 name='FedAvg',
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
                 *args,
                 **kwargs):
        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, update_client, write_log)
    def run(self):
        training_nums = self.send_require_attr('local_training_number')
        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            self.send_train_order(self.epochs)
            w_locals, _ = self.send_require_training_result()
            w_global = self.aggregate(w_locals, training_nums)  
            self.model.load_state_dict(w_global)
            self.current_training_num += self.epochs * batch_num
    @staticmethod
    def aggregate(w_locals, training_nums):
        training_num = sum(training_nums)
        averaged_params = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number = training_nums[i]
                local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
