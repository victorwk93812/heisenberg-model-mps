import copy
from typing import Tuple
import numpy as np

def random_unit_vec(n: int, use_complex: bool = True):
    real = np.random.randn(n)
    if use_complex:
        img = np.random.randn(n)
        v = real + 1j * img
    else:
        v = real
    return v / np.linalg.norm(v)

def random_unitary(n, m):
    trans = False
    if n < m:
        n, m = m, n
        trans = True

    # Random complex matrix: real + i·imag
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)

    # QR decomposition
    q, r = np.linalg.qr(z)

    # Normalize phases to ensure unitarity
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = q * ph

    if trans:
        return q[:,0:m].T
    return q[:,0:m]

def lanczos(Hmat):
    """Returns a unitary matrix and diagonal and superdiagonal entries of tridiagonal matrix.

    !!!Practice usage, not working!!!
    Performs Lanczos algorithm on Hermitian matrix "Hmat" to obtain orthonormal basis V
    and tridiagonal matrix with diagonal entries alpha and superdiagonal entries beta
    """
    Hmat = np.array(Hmat)
    if Hmat.ndim != 2 or Hmat.shape[0] != Hmat.shape[1]:
        raise ValueError(f"Cannot perform Lanczos on {repr(Hmat)}.")

    n = Hmat.shape[0]
    V = np.zeros((n, n), dtype='complex128') # Lanczos basis
    alpha = np.zeros(n, dtype='complex128') # diagonal entries of tridiagonal mat.
    beta = np.zeros(n, dtype='complex128') # superdiagonal entries or tridiagonal mat.

    # First unit vec. and norm
    q = random_unit_vec(n)
    V[:][0] = np.copy(q)
    r = Hmat @ q
    alpha[0] = np.conjugate(q) @ r
    r = r - alpha[0] * q
    beta[0] = np.linalg.norm(r)

    # Iteration to obtain orthonormal basis and norms
    for j in range(1, n):
        v = np.copy(q)
        q = r / beta[j - 1]
        V[:][j] = np.copy(q)
        r = Hmat @ q - beta[j - 1] @ v
        alpha[j] = np.conjugate(q) @ r
        r = r - alpha[j] * q
        beta[j] = np.linalg.norm(r)

class Tensor:
    def __init__(self, value = None):
        if value is None:
            self.value = np.array([])
        else:
            self.value = np.array(value)
        self.shape = np.array(self.value.shape) # will be [0] if self.value is nothing "[]"
        self.size = np.prod(self.shape)

    def __str__(self):
        return f""" Tensor value: {self.value} 
Tensor shape: {self.shape} 
Tensor size: {self.size}"""

    def __repr__(self):
        return f"{self.value}"

    def __getitem__(self, key):
        sliced_arr = self.value[key]  # delegate to the NumPy array
        return Tensor(sliced_arr)

    def __setitem__(self, key, num):
        self.value[key] = num

    def reshape(self, shape):
        shape = np.array(shape)
        if np.prod(shape) != self.size:
            raise ValueError(f"Cannot reshape array of size {self.size} into shape {tuple(shape)}")
        self.value = self.value.reshape(shape)
        self.shape = shape

    def transpose(self, axes = None):
        if axes is None:
            axes = [0]
        self.value = self.value.transpose(axes)
        print(axes)
        self.shape = np.array([self.shape[j] for j in axes])

    @classmethod
    def conjugate(cls, t: 'Tensor', *args, **kwargs):
        return Tensor(np.conjugate(t.value, *args, **kwargs))

    @staticmethod
    def einsum(subscripts, *operands, **kwargs) -> 'Tensor':
        """Wrapper of np.einsum with ndarray replaced with Tensor operands."""
        arroperands = [ten.value for ten in operands]
        return Tensor(np.einsum(subscripts, *arroperands, **kwargs))

    @staticmethod
    def full_svd(a, full_s = True, **kwargs) -> Tuple['Tensor', 'Tensor', 'Tensor']:
        """Wrapper of np.linalg.svd with ndarray replaced with Tensor operands."""
        U, S, Vh = np.linalg.svd(a.value, full_matrices = True, **kwargs)
        if full_s:
            m, n = U.shape[0], Vh.shape[0]
            s_mat = np.zeros((m, n))
            k = min(m, n)
            s_mat[:k, :k] = np.diag(S)
            S = s_mat
        return Tensor(U), Tensor(S), Tensor(Vh)














