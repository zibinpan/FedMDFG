import torch
import numpy as np
import cvxopt
from cvxopt import matrix
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = 0.5 * (P + P.T)  
    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [matrix(P), matrix(q)]  
    if G is not None:
        args.extend([matrix(G), matrix(h)])  
        if A is not None:
            args.extend([matrix(A), matrix(b)])  
    sol = cvxopt.solvers.qp(*args)
    optimal_flag = 1
    if 'optimal' not in sol['status']:
        optimal_flag = 0
    return np.array(sol['x']).reshape((P.shape[1],)), optimal_flag
def setup_qp_and_solve(vec):
    P = np.dot(vec, vec.T)
    n = P.shape[0]
    q = np.zeros(n)
    G = - np.eye(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    cvxopt.solvers.options['show_progress'] = False
    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag
def setup_qp_and_solve_for_mgdaplus(vec, epsilon, lambda0):
    P = np.dot(vec, vec.T)
    n = P.shape[0]
    q = np.zeros(n)
    G = np.vstack([-np.eye(n), np.eye(n)])
    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    h = np.hstack([lb, ub])
    A = np.ones((1, n))
    b = np.ones(1)
    cvxopt.solvers.options['show_progress'] = False
    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag
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
def setup_qp_and_solve_for_mgdaplus_1(vec, epsilon, lambda0):
    P = np.dot(vec, vec.T)
    n = P.shape[0]
    q = np.array([[0] for i in range(n)])
    A = np.ones(n).T
    b = np.array([1])
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
    return sol, 1
def get_d_moomtl_d(grads, device):
    vec = grads
    sol, optimal_flag = setup_qp_and_solve(vec.cpu().detach().numpy())  
    sol = torch.from_numpy(sol).float().to(device)
    d = torch.matmul(sol, grads)
    descent_flag = 1
    c = - (grads @ d)
    if not torch.all(c <= 1e-6):
        descent_flag = 0
    return d, optimal_flag, descent_flag
def get_d_mgdaplus_d(grads, device, epsilon, lambda0):
    vec = grads
    sol, optimal_flag = setup_qp_and_solve_for_mgdaplus_1(vec.cpu().detach().numpy(), epsilon, lambda0)
    sol = torch.from_numpy(sol).float().to(device)
    d = torch.matmul(sol, grads)
    descent_flag = 1
    c = -(grads @ d)
    if not torch.all(c <= 1e-6):
        descent_flag = 0
    return d, optimal_flag, descent_flag
def get_fedmdfg_d(grads, value, add_grads, alpha, fair_guidance_vec, force_active, device):
    fair_grad = None
    value_norm = torch.norm(value)
    norm_values = value / value_norm
    fair_guidance_vec /= torch.norm(fair_guidance_vec)
    cos = float(norm_values @ fair_guidance_vec)
    cos = min(1, cos)
    cos = max(-1, cos)
    bias = np.arccos(cos) / np.pi * 180
    pref_active_flag = (bias > alpha) | force_active
    norm_vec = torch.norm(grads, dim=1)
    indices = list(range(len(norm_vec)))
    grads = norm_vec[indices].reshape(-1, 1) * grads / (norm_vec + 1e-6).reshape(-1, 1)
    if not pref_active_flag:
        vec = grads
        pref_active_flag = 0
    else:
        pref_active_flag = 1
        h_vec = (fair_guidance_vec @ norm_values * norm_values - fair_guidance_vec).reshape(1, -1)
        h_vec /= torch.norm(h_vec)
        fair_grad = h_vec @ grads
        vec = torch.cat((grads, fair_grad))
    if add_grads is not None:
        norm_vec = torch.norm(add_grads, dim=1)
        indices = list(range(len(norm_vec)))
        random.shuffle(indices)
        add_grads = norm_vec[indices].reshape(-1, 1) * add_grads / (norm_vec + 1e-6).reshape(-1, 1)
        vec = torch.vstack([vec, add_grads])
    sol, _ = setup_qp_and_solve(vec.cpu().detach().numpy())
    sol = torch.from_numpy(sol).float().to(device)
    d = sol @ vec
    return d, vec, pref_active_flag, fair_grad
