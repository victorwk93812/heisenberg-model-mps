import numpy as np
from typing import List, Tuple, Final, Literal
import matplotlib.pyplot as plt

from tensor_class import Tensor
from stopwatch import stopwatch

from rich.progress import track

N = 20
d: Final[int] = 2
D = 64
J = 1
g = 0.1
sweep_times = 3
target_var = -100

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

def calc_energy(MPS, MPO, MPS_DAG):
    Energy: Tensor = MPS[0]
    Energy = Energy * MPO[0]
    Energy = Energy * MPS_DAG[0]
    for i in range(1, N):
        Energy = Energy * MPS[i]
        Energy = Energy * MPO[i]
        Energy = Energy * MPS_DAG[i]
    return Energy.value / N
def calc_energy_variance(MPS, MPO, MPO_SQ, MPS_DAG):
    Energy_sq: Tensor = MPS[0] * MPO_SQ[0] * MPS_DAG[0]
    for i in range(1, N):
        Energy_sq = Energy_sq * MPS[i] * MPO_SQ[i] * MPS_DAG[i]
    Energy: Tensor = MPS[0] * MPO[0] * MPS_DAG[0]
    for i in range(1, N):
        Energy = Energy * MPS[i] * MPO[i] * MPS_DAG[i]
    return Energy_sq.value - Energy.value ** 2

def main():
    sw = stopwatch()
    # init MPS
    MPS: List[Tensor] = []
    MPS_DAG: List[Tensor] = []

    init_vurtial_dim = [min(d**(i+1), D, d**(N-i-1)) for i in range(N-1)]
    # init_vurtial_dim = [D] * (N-1)

    bonds = ['s0', 'a1']
    value = random_unitary(d, init_vurtial_dim[0])
    tensor = Tensor(bonds, value)
    MPS.append(tensor)
    MPS_DAG.append(tensor.conj())

    for i in range(1, N-1):
        value = random_unitary(init_vurtial_dim[i-1] * d, init_vurtial_dim[i])
        value = value.reshape(init_vurtial_dim[i-1], d, init_vurtial_dim[i])
        bonds = [f'a{i}', f's{i}', f'a{i+1}']
        tensor = Tensor(bonds, value)
        MPS.append(tensor)
        MPS_DAG.append(tensor.conj())

    bonds = [f'a{N-1}', f's{N-1}']
    value = random_unitary(init_vurtial_dim[N-2], d)
    tensor = Tensor(bonds, value)
    MPS.append(tensor)
    MPS_DAG.append(tensor.conj())

    # init MPO
    MPO: List[Tensor] = []
    W = np.array([
        [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
        [1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0],
        [0, g, g, 0], [J, 0, 0, -J], [1, 0, 0, 1]
    ]).reshape(3, 3, 2, 2)
    w0 = np.tensordot(np.array([0, 0, 1]), W, axes=([0], [0]))
    wN = np.tensordot(W, np.array([1, 0, 0]), axes=([1], [0])) # N-1

    MPO.append(Tensor(['b1', 's0', 's0p'], w0))
    for i in range(1, N-1):
        MPO.append(Tensor([f'b{i}', f'b{i+1}', f's{i}', f's{i}p'], W))
    MPO.append(Tensor([f'b{N-1}', f's{N-1}', f's{N-1}p'], wN))

    MPO_SQ: List[Tensor] = []
    temp1 = Tensor(['b1', 's0', 'a'], w0)
    temp2 = Tensor(['b1p', 'a', 's0p'], w0)
    MPO_SQ.append(temp1 * temp2)
    for i in range(1, N-1):
        temp1 = Tensor([f'b{i}', f'b{i+1}', f's{i}', f'a'], W)
        temp2 = Tensor([f'b{i}p', f'b{i+1}p', f'a', f's{i}p'], W)
        MPO_SQ.append(temp1 * temp2)
    temp1 = Tensor([f'b{N-1}', f's{N-1}', 'a'], wN)
    temp2 = Tensor([f'b{N-1}p', 'a', f's{N-1}p'], wN)
    MPO_SQ.append(temp1 * temp2)

    # init L, R
    L_stack: List[Tensor] = []  # save N-1 item
    R_stack: List[Tensor] = []
    L_init_cursor = MPS[0]
    L_init_cursor = L_init_cursor * MPO[0]
    L_init_cursor = L_init_cursor * MPS_DAG[0]
    L_stack.append(L_init_cursor)
    for i in range(1, N-1):
        L_init_cursor = L_init_cursor * MPS[i]
        L_init_cursor = L_init_cursor * MPO[i]
        L_init_cursor = L_init_cursor * MPS_DAG[i]
        L_stack.append(L_init_cursor)

    sweep_cursor = N - 1
    go_left = True
    energy_list = []
    var_list = []
    label_list = []

    energy = calc_energy(MPS, MPO, MPS_DAG)
    var = calc_energy_variance(MPS, MPO, MPO_SQ, MPS_DAG)
    energy_list.append(energy)
    var_list.append(var)
    label_list.append(sweep_cursor)
    print(f"Initial\n"
          f"Energy: {energy_list[-1]}\n"
          f"Variance: {var_list[-1]}")
    sw.lap("init")

    for i in track(range(2 * sweep_times * N), description="Sweeping..."):
        print(f"step {i}")
        if go_left == True:
            if sweep_cursor == 0:
                go_left = False
                continue

            if R_stack == []:
                H_eff =  L_stack[-1] * MPO[sweep_cursor]
            else:
                H_eff =  L_stack[-1] * MPO[sweep_cursor] * R_stack[-1]
            sw.lap("construct H_eff")

            find_bonds = [b for b in H_eff.bonds if not b.endswith('p')]
            eigen_state = H_eff.ground_state(find_bonds)
            sw.lap("find gs")

            MPS[sweep_cursor] = eigen_state
            MPS_DAG[sweep_cursor] = eigen_state.conj()

            energy = calc_energy(MPS, MPO, MPS_DAG)
            var = calc_energy_variance(MPS, MPO, MPO_SQ, MPS_DAG)
            energy_list.append(energy)
            var_list.append(var)
            label_list.append(sweep_cursor)
            if var < target_var: break
            sw.lap("calc E, var")
            
            _, _, vh = eigen_state.svd(combine_bonds=[f'a{sweep_cursor}'],
                                        type="first",
                                        vh_new_bond_name=f'a{sweep_cursor}')
            MPS[sweep_cursor] = vh
            MPS_DAG[sweep_cursor] = vh.conj()
            sw.lap("canonicalize (svd)")

            if R_stack == []:
                R_stack.append(MPS[sweep_cursor] * MPO[sweep_cursor] * MPS_DAG[sweep_cursor])
            else:
                R_stack.append(R_stack[-1] * MPS[sweep_cursor] * MPO[sweep_cursor] * MPS_DAG[sweep_cursor])
            L_stack.pop()
            sweep_cursor -= 1
        else:
            if sweep_cursor == N-1:
                go_left = True
                print(f"Round: {int(i / N / 2) + 1}\n"
                      f"Energy: {energy_list[-1]}\n"
                      f"Variance: {var_list[-1]}")
                continue

            if L_stack == []:
                H_eff = MPO[sweep_cursor] * R_stack[-1]
            else:
                H_eff = L_stack[-1] * MPO[sweep_cursor] * R_stack[-1]
            sw.lap("construct H_eff")

            find_bonds = [b for b in H_eff.bonds if not b.endswith('p')]
            eigen_state = H_eff.ground_state(find_bonds)
            sw.lap("find gs")

            MPS[sweep_cursor] = eigen_state
            MPS_DAG[sweep_cursor] = eigen_state.conj()

            energy = calc_energy(MPS, MPO, MPS_DAG)
            var = calc_energy_variance(MPS, MPO, MPO_SQ, MPS_DAG)
            energy_list.append(energy)
            var_list.append(var)
            label_list.append(sweep_cursor)
            if var < target_var: break
            sw.lap("calc E, var")

            u, _, _ = eigen_state.svd(combine_bonds=[f'a{sweep_cursor+1}'],
                                        type="second",
                                        u_new_bond_name=f'a{sweep_cursor+1}')
            MPS[sweep_cursor] = u
            MPS_DAG[sweep_cursor] = u.conj()
            sw.lap("canonicalize (svd)")
            
            if L_stack == []:
                L_stack.append(MPS[sweep_cursor] * MPO[sweep_cursor] * MPS_DAG[sweep_cursor])
            else:
                L_stack.append(L_stack[-1] * MPS[sweep_cursor] * MPO[sweep_cursor] * MPS_DAG[sweep_cursor])
            R_stack.pop()
            sweep_cursor += 1

    print("Final Result\n"
            f"Energy per Site: {energy_list[-1]}\n"
            f"Variance: {var_list[-1]}")

    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(label_list, energy_list, )
    ax[1][0].plot(label_list, var_list)
    ax[0][1].plot(energy_list)
    ax[1][1].plot(var_list)
    plt.show()




if __name__ == '__main__':
    main()