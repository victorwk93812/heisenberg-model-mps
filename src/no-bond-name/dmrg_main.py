import numpy as np
from scipy.sparse.linalg import eigsh
from dmrg_utils import random_unitary, Tensor
from dmrg_exceptions import *

N = 10
D = 16
d = 2
g = 1
J = -1
id2 = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
t = 3

def MPS_rand_init(d: int, D: int, N: int):
    MPS, MPSh = [0] * N, [0] * N
    i, lvbdim = 0, 1 # left virtual bond dimension

    # vbonds dim = 1, d, d**2, ...
    while i < N and lvbdim * d <= D and lvbdim * d <= (d**(N - i - 1)):
        MPS[i] = Tensor(random_unitary(lvbdim * d, lvbdim * d))
        MPS[i].reshape([lvbdim, d, lvbdim * d])
        MPSh[i] = Tensor.conjugate(MPS[i])
        i += 1
        lvbdim *= d

    # D <= d**(N - i - 1): all vbonds truncate to dim D
    # loop will run until D > d**(N - i - 1)
    while lvbdim * d <= (d**(N - i - 1)): 
        MPS[i] = Tensor(random_unitary(lvbdim * d, D))
        MPS[i].reshape([lvbdim, d, D])
        MPSh[i] = Tensor.conjugate(MPS[i])
        i += 1
        lvbdim = D
    
    # vbonds dim = lvbdim, d**(N - i - 1), d**(N - i - 2), ..., d, 1
    rvbdim = d**(N - i - 1)
    while i < N:
        MPS[i] = Tensor(random_unitary(lvbdim * d, rvbdim))
        MPS[i].reshape([lvbdim, d, rvbdim])
        MPSh[i] = Tensor.conjugate(MPS[i])
        i += 1
        lvbdim = rvbdim
        rvbdim //= d
    return MPS, MPSh

def MPO_hsnbg_init(g, J, N):
    """Heisenberg model MPO array"""
    MPO = [0] * N
    MPO[0] = Tensor([[-g * sx, -J * sz, np.eye(2)]])
    for i in range(1, N - 1):
        MPO[i] = Tensor([
            [np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))],
            [sz, np.zeros((2, 2)), np.zeros((2, 2))],
            [-g * sx, -J * sz, np.eye(2)]
            ])
    MPO[N - 1] = Tensor([[np.eye(2)], [sz], [-g * sx]])
    return MPO

def sweep_left(MPS, MPO, MPSh, pre, suf):
    return 0

def sweep_right(MPS, MPO, MPSh, pre, suf):
    return 0

# Initialize random left-cnc MPS and MPS conj-trans(MPSh)
# MPS, MPSh are 0-based
# Bond order in MPS[i].shape: lv, p, rv
# Bond order in MPSh[i].shape: lv, p, rv
MPS, MPSh = MPS_rand_init(d, D, N) 
# print(MPS[0], '\n')
# print(MPS[4].shape, '\n')
# print(MPS[9], '\n')

# Initialize MPO
# Bond order in MPO[i].shape: lv, rv, dp, up
# MPO is 0-based
MPO = MPO_hsnbg_init(g, J, N)
# print(MPO[0])
# print(MPO[1])
# print(MPO[N - 1])

# Calc pre[i] (L_i) & declare suf[i] (R_i)
pre, suf = [0] * (N + 1), [0] * (N + 1)
pre[0], suf[0] = Tensor([[[1]]]), Tensor([[[1]]])
for i in range(1, N + 1):
    pre[i] = Tensor.einsum('abc,adi,bjed,cek->ijk', pre[i - 1], MPS[i - 1], MPO[i - 1], MPSh[i - 1])
E0 = Tensor.einsum('iii->', pre[N]).value
print(f"E0: {E0}", type(E0))

# t routines: sweep left -> right -> calc E
El, Er = [], [] # energy after left, right sweeping
for i in range(t):
    El.append(sweep_left(MPS, MPO, MPSh, pre, suf))
    Er.append(sweep_right(MPS, MPO, MPSh, pre, suf))
    print("Energy after left sweep {i}: {El[i]}")
    print("Energy after right sweep {i}: {Er[i]}")

# Tensor class tests
# print("===tensor class test===")
# a = Tensor(np.arange(36).reshape(6, 2, 3))
# a.reshape([3, 2, 6])
# a.transpose((2, 0, 1))
# print(f"a:\n{a}")
# print(f"repr(a):\n{repr(a)}")
# print(f"type(a):\n{type(a)}")
# a = a[:4, :, :]
# print(f"a:\n{a}")
# print(f"repr(a):\n{repr(a)}")
# print(f"type(a):\n{type(a)}")
# b = Tensor([])
# print(f"b:\n{b}")
# print(f"repr(b):\n{repr(b)}")
# print(f"type(b):\n{type(b)}")
# c = Tensor(np.arange(24).reshape(4, 6))
# tq, tr = Tensor.qr(c)
# print(f"tq:\n{tq}")
# print(f"repr(tq):\n{repr(tq)}")
# print(f"type(tq):\n{type(tq)}")
# print(f"tr:\n{tr}")
# print(f"repr(tr):\n{repr(tr)}")
# print(f"type(tr):\n{type(tr)}")
# Tensor class tests #2
# a = Tensor(np.arange(6).reshape(3, 2))
# b = Tensor(np.arange(12).reshape(4, 3))
# c = Tensor.einsum('ki, jk->ij', a, b)
# print(f"c:\n{c}")
# print(f"repr(c):\n{repr(c)}")
# print(f"type(c):\n{type(c)}")

# random_unitary tests
# print("===random unitary test===")
# un_array = random_unitary(2, 1)
# print(un_array, un_array.shape)
# mat = np.matrix(un_array)
# print(un_array)
# print(mat, "\n")
# print(np.matmul(mat.H, mat), '\n')
# print(np.matmul(mat, mat.H), '\n')

# conjugate test
# print("===conjugate test===")
# a = Tensor(np.eye(2) + 1j * np.eye(2))
# print(f"a:\n{a}")
# print(f"repr(a):\n{repr(a)}")
# print(f"type(a):\n{type(a)}")
# b = Tensor.conjugate(a)
# print(f"b:\n{b}")
# print(f"repr(b):\n{repr(b)}")
# print(f"type(b):\n{type(b)}")

# svd test
# print("===SVD test===")
# a = Tensor([[3, 2, 2], [2, 3, -2]])
# u, s, vh = Tensor.full_svd(a)
# print(f"a:\n{a}")
# print(f"repr(a):\n{repr(a)}")
# print(f"type(a):\n{type(a)}")
# print(f"u:\n{u}")
# print(f"repr(u):\n{repr(u)}")
# print(f"type(u):\n{type(u)}")
# print(f"s:\n{s}")
# print(f"repr(s):\n{repr(s)}")
# print(f"type(s):\n{type(s)}")
# print(f"vh:\n{vh}")
# print(f"repr(vh):\n{repr(vh)}")
# print(f"type(vh):\n{type(vh)}")

# MPS unitarity test
print("===MPS unitarity test===")
MPS_norm = [0] * (N + 1)
MPS_norm[0] = Tensor([[1]])
for i in range(1, N + 1):
    MPS_norm[i] = Tensor.einsum('ab,aci,bcj->ij', MPS_norm[i - 1], MPS[i - 1], MPSh[i - 1])
print(MPS_norm[0])
print(MPS_norm[10])

