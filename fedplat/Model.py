
import torch
import copy
from torch.autograd import Variable
class Model(torch.nn.Module):
    def __init__(self, device, *args, **kwargs):
        super(Model, self).__init__()
        self.name = 'Model'
        self.device = device
        self.input_require_shape = None
        self.target_require_shape = None
        self.Loc_reshape_list = None
    def generate_net(self, *args, **kwargs):
        raise NotImplementedError
    def forward(self, x):
        raise NotImplementedError
    def create_Loc_reshape_list(self):
        currentIdx = 0
        self.Loc_reshape_list = []
        for i, p in enumerate(self.parameters()):
            flat = p.data.clone().flatten()
            self.Loc_reshape_list.append(torch.arange(currentIdx, currentIdx + len(flat), 1).reshape(p.data.shape))  
            currentIdx += len(flat)
    @staticmethod
    def change_data_device(data, device):
        new_data = None
        if type(data) == torch.Tensor:
            new_data = data.to(device)
        elif type(data) == tuple:
            new_data = []
            for item in data:
                item = item.to(device)
                new_data.append(item)
            new_data = tuple(new_data)
        elif type(data) == list:
            new_data = []
            for item in data:
                item = item.to(device)
                new_data.append(item)
        return new_data
    def __add__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if not isinstance(other, Model):
            raise TypeError
        res_model = copy.deepcopy(self)
        res_model.zero_grad()
        res_model_params = res_model.state_dict()
        other_params = other.state_dict()
        for layer in res_model_params.keys():
            res_model_params[layer] += other_params[layer]
        return res_model
    def __sub__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if not isinstance(other, Model):
            raise TypeError
        res_model = copy.deepcopy(self)
        res_model.zero_grad()
        res_model_params = res_model.state_dict()
        other_params = other.state_dict()
        for layer in res_model_params.keys():
            res_model_params[layer] -= other_params[layer]
        return res_model
    def __mul__(self, other):
        res_model = copy.deepcopy(self)
        res_model.zero_grad()
        res_model_params = res_model.state_dict()
        if not isinstance(other, Model):
            for k in res_model_params.keys():
                res_model_params[k] *= other
        else:
            other_params = other.state_dict()
            for k in res_model.state_dict().keys():
                res_model_params[k] *= other_params[k]
        return res_model
    def __rmul__(self, other):
        return self * other
    def __pow__(self, power):
        return self._model_norm(power)
    def dot(self, other):
        res = 0.0
        md1 = self.state_dict()
        md2 = other.state_dict()
        for layer in md1.keys():
            if md1[layer] is None:
                continue
            res += torch.sum(md1[layer] * md2[layer])
        return res
    def L2_norm_square(self):
        model_params = self.state_dict()
        res = 0.0
        for k in model_params.keys():
            if model_params[k] is None: continue
            if model_params[k].dtype not in [torch.float, torch.float32, torch.float64]:
                continue
            res += torch.sum(torch.pow(model_params[k], 2))
        return res
    def norm(self, p=2):
        return self**p
    def _model_norm(self, p):
        res = 0.0
        md = self.state_dict()
        for layer in md.keys():
            if md[layer] is None:
                continue
            if md[layer].dtype not in [torch.float, torch.float32, torch.float64]:
                continue
            res += torch.sum(torch.pow(md[layer], p))
        return torch.pow(res, 1.0 / p)
    @staticmethod
    def model_sum(model_list):
        res_model = copy.deepcopy(model_list[0])
        res_model_params = res_model.state_dict()
        for model in model_list[1:]:
            model_params = model.state_dict()
            for k in model_params.keys():
                res_model_params[k] += model_params[k]
        return res_model
    @staticmethod
    def model_average(ms, weights=None):
        if weights is None:
            weights = [1.0 / len(ms)] * len(ms)
        res = copy.deepcopy(ms[0])
        res.zero_grad()
        w_locals = [ms[i].state_dict() for i in range(len(ms))]
        averaged_params = res.state_dict()
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_model_params = w_locals[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] * weights[i]
                else:
                    averaged_params[k] += local_model_params[k] * weights[i]
        res.load_state_dict(averaged_params)
        return res
    def span_model_grad_to_vec(self):
        grad_vec = []
        for p in self.parameters():
            if p.grad is not None:
                flat = p.grad.data.clone().flatten()
                grad_vec.append(Variable(flat, requires_grad=False))
        grad_vec = torch.cat(grad_vec)
        return grad_vec
    def span_model_params_to_vec(self):
        param_vec = []
        model_params = self.state_dict()
        for layer in model_params.keys():
            flat = model_params[layer].clone().flatten()
            param_vec.append(Variable(flat, requires_grad=False))
        param_vec = torch.cat(param_vec)
        return param_vec
