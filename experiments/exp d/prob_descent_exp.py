from cal_d import get_FedMGDP_d, get_d_mgdaplus_d, get_MGDA_d, get_FedFV_d
import torch
import copy
import numpy as np
torch.set_default_dtype(torch.float64)

device = 'cpu'

N = 1000

m_list = [2, 3, 10, 100, 100]
D_list = [2, 3, 100, 10, 1000]
idx = list(range(len(m_list)))
result_mat = np.zeros((3, len(m_list)))
m_D_zip = zip(idx, m_list, D_list)


for j, m, D in m_D_zip:
    print(j, ' ', m, ' ', D)
    l_locals = torch.rand(m) * 100

    # loop N times
    count1 = 0
    count2 = 0
    count3 = 0
    for _ in range(N):
        # generate grad_mat
        grad_mat = []
        for _ in range(m):
            g = torch.rand(D) - 0.5
            grad_mat.append(g)
        grad_mat = torch.stack(grad_mat).to(device)

        # FedMGDA+
        epsilon = 0.001
        lambda0 = np.array([1 / m] * m)
        d, descent_flag = get_d_mgdaplus_d(copy.deepcopy(grad_mat/torch.norm(grad_mat, dim=1).reshape(-1, 1)), epsilon, lambda0, device)
        count1 += descent_flag

        # FedFV
        alpha = 0
        d, descent_flag = get_FedFV_d(copy.deepcopy(grad_mat), l_locals, alpha, device)
        count2 += descent_flag

        # FedMGDP
        alpha = 0.25
        p = torch.Tensor([1.0] * m)
        p /= torch.norm(p)
        d, vec, pref_active_flag, fair_grad, descent_flag = get_FedMGDP_d(copy.deepcopy(grad_mat), l_locals, None, alpha, p, False, device)
        # count3 += descent_flag
        weights = torch.Tensor([1 / m] * m).to(device)
        d_r = weights @ grad_mat
        count3 += int(torch.norm(d) <= torch.norm(d_r))
    # calculate the results
    result_mat[0, j] = count1 / N
    result_mat[1, j] = count2 / N
    result_mat[2, j] = count3 / N

# store the results
print(result_mat)
np.savetxt('exp1.csv', result_mat, delimiter=',')
