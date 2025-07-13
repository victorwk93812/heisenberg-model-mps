import numpy as np

# init func
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
# def random_unitary(n, m):
#     return np.eye(np.max([n, m]))[0:n, 0:m]

O = np.zeros((2, 2))
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
S_p = np.array([[0, 1], [0, 0]])
S_m = np.array([[0, 0], [1, 0]])