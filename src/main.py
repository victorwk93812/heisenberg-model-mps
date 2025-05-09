import numpy as np


class DimDismatchError(Exception):
    def __init__(self, dim1, dim2):
        message = f"Values are unequal: {dim1} and {dim2}"
        super().__init__(message)
        self.dim1 = dim1
        self.dim2 = dim2


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


class Tensor:
    all_bond = {}

    def __init__(self, bonds: dict[str, int] = {}, value = None):
        self.bonds = bonds.keys()
        if value == None:
            self.value = []

        for bond in bonds.keys():
            if bond in Tensor.all_bond:
                try:
                    if Tensor.all_bond[bond] != bonds[bond]:
                        raise DimDismatchError(Tensor.all_bond[bond], bonds[bond])
                except DimDismatchError as e:
                    print("Dimensions dismatch!:(")
                    print(f"Expected {e.dim1} but found {e.dim2}")
                    raise
            else:
                Tensor.all_bond[bond] = bonds[bond]

    def set_random_MPS_site(self):
        pass

a = Tensor({'s1': 2, 's2': 3})
b = Tensor({'s2': 3, 's3': 4, 's4': 7})
print(Tensor.all_bond)

un_array = random_unitary(4, 3)
print(un_array[3])
mat = np.mat(un_array)
print(un_array)
print(mat, "\n")
print(np.matmul(mat.H, mat), '\n')
print(np.matmul(mat, mat.H), '\n')
