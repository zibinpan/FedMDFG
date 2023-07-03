
import fedplat as fp
import torch
import copy
from torch.autograd import Variable
class Client:
    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 dishonest=None,
                 *args,
                 **kwargs):
        self.id = id
        if model is not None:
            model = model
        self.model = model
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        self.train_setting = train_setting
        self.metric_list = metric_list
        self.dishonest = dishonest
        self.local_training_data = None
        self.local_training_number = 0
        self.local_test_data = None
        self.local_test_number = 0
        self.training_batch_num = 0
        self.test_batch_num = 0
        self.metric_history = {'training_loss': [],
                               'test_loss': [],
                               'local_test_number': 0}
        for metric in self.metric_list:
            self.metric_history[metric.name] = []  
            if metric.name == 'correct':
                self.metric_history['test_accuracy'] = []  
        self.model_weights = None  
        self.model_loss = None  
        self.info_msg = {}  
        self.initial_lr = float(train_setting['optimizer'].defaults['lr'])
        self.lr = self.initial_lr
        self.optimizer = train_setting['optimizer'].__class__(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.optimizer.defaults = copy.deepcopy(train_setting['optimizer'].defaults)
        self.criterion = self.train_setting['criterion'].to(self.device)
        self.old_model = copy.deepcopy(self.model)
    def update_data(self, id, local_training_data, local_training_number, local_test_data, local_test_number):
        self.id = id
        self.local_training_data = local_training_data
        self.local_training_number = local_training_number
        self.local_test_data = local_test_data
        self.local_test_number = local_test_number
        self.training_batch_num = len(local_training_data)
        self.test_batch_num = len(local_test_data)
    def get_message(self, msg):
        return_msg = {}
        if msg['command'] == 'sync':
            self.model_weights = msg['w_global']
            self.model.load_state_dict(self.model_weights)
            if self.dishonest is not None:
                if self.dishonest['grad norm'] is not None or self.dishonest['inverse grad'] is not None or self.dishonest['random grad'] is not None or self.dishonest['random grad 10'] is not None or self.dishonest['gaussian'] is not None:
                    self.old_model.load_state_dict(copy.deepcopy(self.model_weights))
        if msg['command'] == 'update_learning_rate':
            current_comm_round = msg['current_comm_round']
            self.lr = self.initial_lr * self.train_setting['lr_decay']**current_comm_round
            self.optimizer = fp.Algorithm.update_learning_rate(self.optimizer, self.lr)  
        if msg['command'] == 'cal_loss':
            batch_idx = msg['batch_idx']
            self.cal_loss(batch_idx)
        if msg['command'] == 'cal_all_batches_loss':
            self.info_msg['common_loss_of_all_batches'] = self.cal_all_batches_loss(self.model)
        if msg['command'] == 'cal_all_batches_gradient_loss':
            self.cal_all_batches_gradient_loss()
        if msg['command'] == 'evaluate':
            batch_idx = msg['batch_idx']
            mode = msg['mode']
            self.evaluate(mode, batch_idx)
        if msg['command'] == 'train':
            epochs = msg['epochs']
            lr = msg['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.train(epochs)
        if msg['command'] == 'test':
            self.test()
        if msg['command'] == 'require_cal_loss_result':
            return_loss = self.model_loss
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_cal_all_batches_loss_result':
            return_loss = self.info_msg['common_loss_of_all_batches']
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_all_batches_gradient_loss_result':
            return_grad = self.info_msg['common_gradient_vec_of_all_batches']
            return_loss = self.info_msg['common_loss_of_all_batches']
            if self.dishonest is not None:
                if self.dishonest['grad norm'] is not None:
                    return_grad *= self.dishonest['grad norm']
                if self.dishonest['zero grad'] is not None:
                    return_grad *= 0.0
                if self.dishonest['random grad'] is not None:
                    n = len(return_grad)
                    r = (torch.rand(n) * 2.0 - 1.0).float().to(self.device)
                    r /= torch.norm(r)
                    return_grad = r * torch.norm(return_grad)
                if self.dishonest['gaussian'] is not None:
                    n = len(return_grad)
                    weights = torch.randn(n).float().to(self.device)
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    grad = old_model_params_span - weights
                    return_grad = grad / torch.norm(grad) * torch.norm(return_grad)
            return_msg['g_local'] = return_grad
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_evaluate_result':
            return_grad = self.model.span_model_grad_to_vec()
            return_loss = self.model_loss
            if self.dishonest is not None:
                if self.dishonest['grad norm'] is not None:
                    return_grad *= self.dishonest['grad norm']
                if self.dishonest['zero grad'] is not None:
                    return_grad *= 0.0
                if self.dishonest['random grad'] is not None:
                    n = len(return_grad)
                    r = (torch.rand(n) * 2.0 - 1.0).float().to(self.device)
                    r /= torch.norm(r)
                    return_grad = r * torch.norm(return_grad)
                if self.dishonest['gaussian'] is not None:
                    n = len(return_grad)
                    weights = torch.randn(n).float().to(self.device)
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    grad = old_model_params_span - weights
                    return_grad = grad / torch.norm(grad) * torch.norm(return_grad)
            return_msg['g_local'] = return_grad
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_client_model':
            if msg['requires_grad'] == 'True':
                return_model = copy.deepcopy(self.model)
                return_loss = self.model_loss
            else:
                with torch.no_grad():
                    return_model = copy.deepcopy(self.model)
                    return_loss = self.model_loss
            if self.dishonest is not None:
                if self.dishonest['grad norm'] is not None:
                    return_model = (return_model - self.old_model) * self.dishonest['grad norm'] + self.old_model
                if self.dishonest['zero grad'] is not None:
                    return_model = copy.deepcopy(self.old_model)
                if self.dishonest['random grad'] is not None:
                    model_params_span = return_model.span_model_params_to_vec()
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    n = len(model_params_span)
                    r = (torch.rand(n) * 2.0 - 1.0).float().to(self.device)
                    r /= torch.norm(r)
                    return_model_params_span = r * torch.norm(model_params_span - old_model_params_span) + old_model_params_span
                    for i, p in enumerate(return_model.parameters()):
                        p.data = return_model_params_span[return_model.Loc_reshape_list[i]]
                if self.dishonest['gaussian'] is not None:
                    model_params_span = return_model.span_model_params_to_vec()
                    n = len(model_params_span)
                    weights = torch.randn(n).float().to(self.device)
                    for i, p in enumerate(return_model.parameters()):
                        p.data = weights[return_model.Loc_reshape_list[i]]
            return_msg['m_local'] = return_model
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_training_result':
            return_model = copy.deepcopy(self.model)
            return_loss = self.model_loss
            if self.dishonest is not None:
                if self.dishonest['grad norm'] is not None:
                    return_model = (return_model - self.old_model) * self.dishonest['grad norm'] + self.old_model
                if self.dishonest['inverse grad'] is not None:
                    return_model = (return_model - self.old_model) * (-1) + self.old_model
                if self.dishonest['zero grad'] is not None:
                    return_model = copy.deepcopy(self.old_model)
                if self.dishonest['random grad'] is not None:
                    model_params_span = return_model.span_model_params_to_vec()
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    n = len(model_params_span)
                    r = (torch.rand(n) * 2.0 - 1.0).float().to(self.device)
                    r /= torch.norm(r)
                    return_model_params_span = r * torch.norm(model_params_span - old_model_params_span) + old_model_params_span
                    for i, p in enumerate(return_model.parameters()):
                        p.data = return_model_params_span[return_model.Loc_reshape_list[i]]
                if self.dishonest['gaussian'] is not None:
                    model_params_span = return_model.span_model_params_to_vec()
                    n = len(model_params_span)
                    weights = torch.randn(n).float().to(self.device)
                    for i, p in enumerate(return_model.parameters()):
                        p.data = weights[return_model.Loc_reshape_list[i]]
            return_msg['w_local'] = return_model.state_dict()
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_test_result':
            return_msg['metric_history'] = copy.deepcopy(self.metric_history)
        if msg['command'] == 'require_attribute_value':
            attr = msg['attr']
            return_msg['attr'] = getattr(self, attr)
        return return_msg
    def cal_loss(self, batch_idx):
        self.model.train()
        with torch.no_grad():
            batch_x, batch_y = self.local_training_data[batch_idx]
            batch_x = self.model.change_data_device(batch_x, self.device)
            batch_y = self.model.change_data_device(batch_y, self.device)
            out = self.model(batch_x)
            loss = self.criterion(out, batch_y)
            self.model_loss = Variable(loss, requires_grad=False)
    def cal_all_batches_loss(self, model):
        model.train()
        total_loss = 0  
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                out = model(batch_x)
                loss = self.criterion(out, batch_y)
                total_loss += loss * batch_y.shape[0]  
            loss = total_loss / self.local_training_number
        return loss
    def update_model(self, model, d, lr):
        self.optimizer = self.train_setting['optimizer'].__class__(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for i, p in enumerate(model.parameters()):
            p.grad = d[self.model.Loc_reshape_list[i]]  
        self.optimizer.step()
        return model
    def cal_all_batches_gradient_loss(self):
        self.model.train()
        grad_mat = []  
        total_loss = 0  
        weights = []
        for step, (batch_x, batch_y) in enumerate(self.local_training_data):
            batch_x = fp.Model.change_data_device(batch_x, self.device)
            batch_y = fp.Model.change_data_device(batch_y, self.device)
            weights.append(batch_y.shape[0])
            out = self.model(batch_x)
            loss = self.criterion(out, batch_y)
            total_loss += loss * batch_y.shape[0]  
            self.model.zero_grad()
            loss.backward()
            grad_vec = self.model.span_model_grad_to_vec()
            grad_mat.append(grad_vec)
        loss = total_loss / self.local_training_number
        weights = torch.Tensor(weights).float().to(self.device)
        weights = weights / torch.sum(weights)
        grad_mat = torch.stack([grad_mat[i] for i in range(len(grad_mat))])
        g = weights @ grad_mat
        self.info_msg['common_gradient_vec_of_all_batches'] = g
        self.info_msg['common_loss_of_all_batches'] = Variable(loss, requires_grad=False)
    def evaluate(self, mode, batch_idx):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise RuntimeError('error in Client: mode can only be train or model')
        batch_x, batch_y = self.local_training_data[batch_idx]
        batch_x = fp.Model.change_data_device(batch_x, self.device)
        batch_y = fp.Model.change_data_device(batch_y, self.device)
        out = self.model(batch_x)
        loss = self.criterion(out, batch_y)
        self.model.zero_grad()
        loss.backward()
        self.model_loss = Variable(loss, requires_grad=False)
    def train(self, epochs):
        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')
        loss = self.cal_all_batches_loss(self.model)
        self.model_loss = Variable(loss, requires_grad=False)
        self.model.train()
        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                out = self.model(batch_x)
                loss = self.criterion(out, batch_y)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
    def test(self):
        self.model.eval()
        criterion = self.train_setting['criterion'].to(self.device)
        metric_dict = {'training_loss': 0, 'test_loss': 0}
        for metric in self.metric_list:
            metric_dict[metric.name] = 0  
            if metric.name == 'correct':
                metric_dict['test_accuracy'] = 0  
        with torch.no_grad():
            for (batch_x, batch_y) in self.local_training_data:
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                out = self.model(batch_x)
                loss = criterion(out, batch_y).item()
                metric_dict['training_loss'] += loss * batch_y.shape[0]  
            self.metric_history['training_loss'].append(metric_dict['training_loss'] / self.local_training_number)
            self.metric_history['local_test_number'] = self.local_test_number
            for (batch_x, batch_y) in self.local_test_data:
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                out = self.model(batch_x)
                loss = criterion(out, batch_y).item()
                metric_dict['test_loss'] += loss * batch_y.shape[0]  
                for metric in self.metric_list:
                    metric_dict[metric.name] += metric.calc(out, batch_y)
            self.metric_history['test_loss'].append(metric_dict['test_loss'] / self.local_test_number)
            for metric in self.metric_list:
                self.metric_history[metric.name].append(metric_dict[metric.name])
                if metric.name == 'correct':
                    self.metric_history['test_accuracy'].append(100 * metric_dict['correct'] / self.local_test_number)
