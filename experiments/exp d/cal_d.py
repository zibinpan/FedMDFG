import torch
import numpy as np
import cvxopt
from cvxopt import matrix
import os
import copy
import math
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):

    P = 0.5 * (P + P.T)  # make sure P is symmetric

    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    return np.array(sol['x']).reshape((P.shape[1],))


def setup_qp_and_solve(vec):
    # use cvxopt to solve QP
    P = np.dot(vec, vec.T)

    n = P.shape[0]
    q = np.zeros(n)

    G = - np.eye(n)
    h = np.zeros(n)

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False

    sol = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol


def get_MGDA_d(grads, device):
    """ calculate the gradient direction for FedMGDA """

    vec = grads
    sol = setup_qp_and_solve(vec.cpu().detach().numpy())  # using CVX to solve the QP problem

    sol = torch.from_numpy(sol).to(device)
    # print('sol: ', sol)
    d = torch.matmul(sol, grads)

    # check descent direction
    descent_flag = 1
    c = - (grads @ d)
    if not torch.all(c <= 1e-6):
        descent_flag = 0

    return d, descent_flag


def quadprog(P, q, G, h, A, b):
    P = cvxopt.matrix(P.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
    return np.array(sol['x'])


def setup_qp_and_solve_for_mgdaplus(vec, epsilon, lambda0):
    # use cvxopt to solve QP
    P = np.dot(vec, vec.T)

    n = P.shape[0]

    q = np.array([[0] for i in range(n)])
    # equality constraint λ∈Δ
    A = np.ones(n).T
    b = np.array([1])
    # boundary
    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    G = np.zeros((2 * n, n))
    for i in range(n):
        G[i][i] = -1
        G[n + i][i] = 1
    h = np.zeros((2 * n, 1))
    for i in range(n):
        h[i] = -lb[i]
        h[n + i] = ub[i]
    sol = quadprog(P, q, G, h, A, b).reshape(-1)

    return sol


def get_d_mgdaplus_d(grads, epsilon, lambda0, device):
    """ calculate the gradient direction for FedMGDA+ """

    vec = grads
    sol = setup_qp_and_solve_for_mgdaplus(vec.cpu().detach().numpy(), epsilon, lambda0)
    # print('sol: ', sol)

    sol = torch.from_numpy(sol).to(device)
    d = torch.matmul(sol, grads)

    # check descent direction
    descent_flag = 1
    c = -(grads @ d)
    if not torch.all(c <= 1e-5):
        descent_flag = 0
    return d, descent_flag


def get_FedFV_d(grads, value, alpha, device):

    grads = [grads[i, :] for i in range(grads.shape[0])]

    # project grads
    order_grads = copy.deepcopy(grads)
    order = [_ for _ in range(len(order_grads))]

    # sort client gradients according to their losses in ascending orders
    tmp = sorted(list(zip(value, order)), key=lambda x: x[0])
    order = [x[1] for x in tmp]

    # keep the original direction for clients with the αm largest losses
    keep_original = []
    if alpha > 0:
        keep_original = order[math.ceil((len(order) - 1) * (1 - alpha)):]

    # calculate g_locals[j].L2_norm_square() first to be more faster.
    g_locals_L2_norm_square_list = []
    for g_local in grads:
        g_locals_L2_norm_square_list.append(torch.norm(g_local)**2)

    # mitigate internal conflicts by iteratively projecting gradients
    for i in range(len(order_grads)):
        if i in keep_original:
            continue
        for j in order:
            if j == i:
                continue
            else:
                # calculate the dot of gi and gj
                dot = grads[j] @ order_grads[i]
                if dot < 0:
                    order_grads[i] = order_grads[i] - dot / g_locals_L2_norm_square_list[j] * grads[j]

    # aggregate projected grads
    weights = torch.Tensor([1 / len(order_grads)] * len(order_grads)).to(device)
    gt = weights @ torch.stack([order_grads[i] for i in range(len(order_grads))])

    # ||gt||=||1/m*Σgi||
    gnorm = torch.norm(weights @ torch.stack([grads[i] for i in range(len(grads))]))
    # check descent direction
    grads = torch.stack(grads)
    c = -(grads @ gt)
    descent_flag = 1
    if not torch.all(c <= 1e-5):
        descent_flag = 0
    return gt, descent_flag

def get_FedMGDP_d(grads, value, add_grads, alpha, fair_guidance_vec, force_active, device):
    """ calculate the gradient direction for FedMGDP """

    fair_grad = None

    value_norm = torch.norm(value)
    norm_values = value/value_norm
    fair_guidance_vec /= torch.norm(fair_guidance_vec)

    m = grads.shape[0]
    weights = torch.Tensor([1 / m] * m).to(device)
    g_norm = torch.norm(weights @ grads)

    # new check active constraints
    cos = float(norm_values @ fair_guidance_vec)
    cos = min(1, cos)  # prevent float error
    cos = max(-1, cos)  # prevent float error
    bias = np.arccos(cos) / np.pi * 180
    # print('bias:', bias)
    pref_active_flag = (bias > alpha) | force_active

    if not pref_active_flag:
        vec = grads
        pref_active_flag = 0

    else:
        pref_active_flag = 1
        h_vec = (norm_values - fair_guidance_vec / torch.norm(fair_guidance_vec)).reshape(1, -1)
        h_vec /= torch.norm(h_vec)
        fair_grad = h_vec @ grads
        vec = torch.cat((grads, fair_grad))

    if add_grads is not None:
        vec = torch.vstack([vec, add_grads])

    sol = setup_qp_and_solve(vec.cpu().detach().numpy())  # using CVX to solve the QP problem
    sol = torch.from_numpy(sol).to(device)
    d = sol @ vec  # get common gradient
    # check constraints
    descent_flag = 1
    c = - (vec @ d)
    if not torch.all(c <= 1e-5):
        descent_flag = 0

    return d, vec, pref_active_flag, fair_grad, descent_flag
