from this import d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(elev=15, azim=-7)

# paste the results generated by algorithms
grad_mat = np.array([[ 2.0000, -0.8000, -0.2000],
        [-1.6000,  2.0000, -0.2000],
        [-1.0000, -1.0000,  1.0000]])
add_grad = np.array([[-2.0149, -0.1610,  0.7659]])
O = np.array([0, 0, 0])
A = grad_mat[0, :]
B = grad_mat[1, :]
C = grad_mat[2, :]

d_all = np.array([[-0.5345713717748627 , -0.049313063055806856 , 0.8436833945804526 , ],
[0.059806403334763454 , 0.06792557930672997 , 0.9958962344521666 , ],
[-0.6882472016116853 , 0.2294157338705617 , 0.6882472016116852 , ],
[0.271337267787339 , 0.3488622631175467 , 0.8970346751838557 , ],
])

d_FedFV = d_all[0, :]
d_FedMGDA_plus = d_all[1, :]
d_FedSGD = d_all[2, :]
d_FedMGDP = d_all[3, :]

# prepare for drawing
d_randoms = np.random.rand(100000, 3)
d_randoms = d_randoms / np.linalg.norm(d_randoms, axis=1).reshape(-1, 1)
c = -(grad_mat @ d_randoms.T)
d_descents = d_randoms[np.where(np.all(c <= 1e-5, axis=0))[0], :]
print(d_descents.shape)
shallow_prepare = np.zeros((0, 3))
for i in range(d_descents.shape[0]):
    shallow_prepare = np.vstack([shallow_prepare, np.zeros((1, 3))])
    shallow_prepare = np.vstack([shallow_prepare, d_descents[i, :]])
common_descent_directions = shallow_prepare.T

new_grad_mat = np.vstack([grad_mat, add_grad])
d_randoms = np.random.rand(100000, 3)
d_randoms = d_randoms / np.linalg.norm(d_randoms, axis=1).reshape(-1, 1)
c = -(new_grad_mat @ d_randoms.T)
d_descents = d_randoms[np.where(np.all(c <= 1e-5, axis=0))[0], :]
print(d_descents.shape)
shallow_prepare = np.zeros((0, 3))
for i in range(d_descents.shape[0]):
    shallow_prepare = np.vstack([shallow_prepare, np.zeros((1, 3))])
    shallow_prepare = np.vstack([shallow_prepare, d_descents[i, :]])
fair_directions = shallow_prepare.T

ax.plot([O[0], A[0]], [O[1], A[1]], [O[2], A[2]], '-', c='black')
ax.plot([O[0], B[0]], [O[1], B[1]], [O[2], B[2]], '-', c='black')
ax.plot([O[0], C[0]], [O[1], C[1]], [O[2], C[2]], '-', c='black')

ax.plot([O[0], d_FedSGD[0]], [O[1], d_FedSGD[1]], [O[2], d_FedSGD[2]], '-', c='red', label='FedSGD')
ax.plot([O[0], d_FedFV[0]], [O[1], d_FedFV[1]], [O[2], d_FedFV[2]], '--', c='red', label='FedFV')
ax.plot([O[0], d_FedMGDA_plus[0]], [O[1], d_FedMGDA_plus[1]], [O[2], d_FedMGDA_plus[2]], ':', c='red', label='FedMGDA+')
ax.plot([O[0], d_FedMGDP[0]], [O[1], d_FedMGDP[1]], [O[2], d_FedMGDP[2]], '-', c='green', label='FedMDFG')

ax.plot(common_descent_directions[0, :], common_descent_directions[1, :], common_descent_directions[2, :], '-', c='gray', alpha=0.5, label='All possible common descent directions')

ax.plot(fair_directions[0, :], fair_directions[1, :], fair_directions[2, :], '-', c='yellow', alpha=0.5, label='All possible fair directions')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.legend()
plt.show()
