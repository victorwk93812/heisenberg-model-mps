import math
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import qr, rq
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

def sweep_left(MPS, MPO, MPSh, pre, suf) -> complex:
    # Update each MPS(h) site with local GS
    for i in range(0, N - 1):
        # Calc. Heff matrix
        Heff = Tensor.einsum(
                'mai,abjn,bcko,pcl->ijklmnop', 
                pre[N - 2 - i], MPO[N - 2 - i], 
                MPO[N - 1 - i], suf[i]).value # Effective Ham. ndarray
        sh = Heff.shape # (i, j, k, l, m, n, o, p)
        matsh = (math.prod(sh[:4]), math.prod(sh[4:]))
        Heff = Heff.reshape(matsh)

        # Find local GS and GSE
        eigvals, eigvecs = eigsh(Heff, k=1, which='SA') # smallest eigval. 
        print(f"Heff local GSE on left sweep site {i} update: {eigvals[0]}")

        # SVD(RQ here) 
        lvbdim, lpbdim, rpbdim, rvbdim = sh[4:8] # merged MPS site dim
        eigvec = eigvecs.reshape((lvbdim * lpbdim, rpbdim * rvbdim)) # before svd/qr/rq dec.
        lsite, rsite = rq(eigvec, mode='economic')

        # Truncation & update MPS(h)
        truncdim = MPS[N - 1 - i].shape[0] # truncated vbond dim after svd/qr/rq
        # print(f"Sweep {i} dims:", lvbdim, lpbdim, truncdim, rpbdim, rvbdim)
        rsite = rsite[:truncdim, :].reshape((truncdim, rpbdim, rvbdim)) # trunc. rows
        MPS[N - 1 - i] = Tensor(rsite)
        MPSh[N - 1 - i] = Tensor.conjugate(MPS[N - 1 - i])
        lsite = lsite[:, :truncdim].reshape((lvbdim, lpbdim, truncdim)) # trunc. cols. 
        MPS[N - 2 - i] = Tensor(lsite)
        MPSh[N - 2 - i] = Tensor.conjugate(MPS[N - 2 - i])

        # Calc. R_{i+1} (suf[i + 1])
        suf[i + 1] = Tensor.einsum('abc,ida,jbed,kec->ijk', 
                                   suf[i], MPS[N - 1 - i], MPO[N - 1 - i], MPSh[N - 1 - i])
        # print(suf[i + 1].shape)
        print(f"MPS site {N - i} shape: {MPS[N - 1 - i].shape}")

    # Calculate GS energy
    suf[N] = Tensor.einsum('abc,ida,jbed,kec->ijk', 
                               suf[N - 1], MPS[0], MPO[0], MPSh[0])
    E = complex(Tensor.einsum('iii->', suf[N]).value)
    return E

def sweep_right(MPS, MPO, MPSh, pre, suf) -> complex:
    # Update each MPS(h) site with local GS
    for i in range(0, N - 1):
        # Calc. Heff matrix
        Heff = Tensor.einsum(
                'mai,abjn,bcko,pcl->ijklmnop', 
                pre[i], MPO[i], 
                MPO[i + 1], suf[N - 2 - i]).value # Effective Ham. ndarray
        sh = Heff.shape # (i, j, k, l, m, n, o, p)
        matsh = (math.prod(sh[:4]), math.prod(sh[4:]))
        Heff = Heff.reshape(matsh)

        # Find local GS and GSE
        eigvals, eigvecs = eigsh(Heff, k=1, which='SA') # smallest eigval. 
        print(f"Heff local GSE on left sweep site {i} update: {eigvals[0]}")

        # SVD(QR here) 
        lvbdim, lpbdim, rpbdim, rvbdim = sh[4:8] # merged MPS site dim
        eigvec = eigvecs.reshape((lvbdim * lpbdim, rpbdim * rvbdim)) # before svd/qr/rq dec.
        lsite, rsite = qr(eigvec, mode='economic')

        # Truncation & update MPS(h)
        truncdim = MPS[i + 1].shape[0] # truncated vbond dim after svd/qr/rq
        # print(f"Sweep {i} dims:", lvbdim, lpbdim, truncdim, rpbdim, rvbdim)
        rsite = rsite[:truncdim, :].reshape((truncdim, rpbdim, rvbdim)) # trunc. rows
        MPS[i + 1] = Tensor(rsite)
        MPSh[i + 1] = Tensor.conjugate(MPS[i + 1])
        lsite = lsite[:, :truncdim].reshape((lvbdim, lpbdim, truncdim)) # trunc. cols. 
        MPS[i] = Tensor(lsite)
        MPSh[i] = Tensor.conjugate(MPS[i])

        # Calc. L_{i+1} (pre[i + 1])
        suf[i + 1] = Tensor.einsum('abc,adi,bjed,cek->ijk', 
                                   pre[i], MPS[i], MPO[i], MPSh[i])
        # print(pre[i + 1].shape)
        print(f"MPS site {i + 1} shape: {MPS[i].shape}")

    # Calculate GS energy
    pre[N] = Tensor.einsum('abc,adi,bjed,cek->ijk', 
                               pre[N - 1], MPS[N - 1], MPO[N - 1], MPSh[N - 1])
    E = complex(Tensor.einsum('iii->', pre[N]).value)
    return E

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
    pre[i] = Tensor.einsum('abc,adi,bjed,cek->ijk', 
                           pre[i - 1], MPS[i - 1], MPO[i - 1], MPSh[i - 1])
    # print(MPS[i - 1].shape)
E0 = complex(Tensor.einsum('iii->', pre[N]).value)
print(f"Energy before sweeping: {E0}\n")

# t routines: sweep left -> right -> calc E
El, Er = [], [] # energy after left, right sweeping
for i in range(t):
    print(f"===Sweep left {i + 1} start===")
    El.append(sweep_left(MPS, MPO, MPSh, pre, suf))
    print(f"===Sweep left {i + 2} end===\n")
    print(f"Energy after left sweep {i + 1}: {El[i].real:.4f}{'+' if El[i].imag >= 0 else ''}{El[i].imag:.4f}j\n")
    print(f"===Sweep right {i + 1} start===")
    Er.append(sweep_right(MPS, MPO, MPSh, pre, suf))
    print(f"===Sweep right {i + 1} end===\n")
    print(f"Energy after right sweep {i + 1}: {Er[i].real:.4f}{'+' if Er[i].imag >= 0 else ''}{Er[i].imag:.4f}j\n")

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

