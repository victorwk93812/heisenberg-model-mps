import numpy as np
from typing import List, Tuple, Final, Literal
import matplotlib.pyplot as plt

from tensor_class import Tensor
from recorder_class import Recorder
from stopwatch import stopwatch
from matrix_gen import random_unitary

from rich.progress import track

N = 10
d: Final[int] = 2
D = 32
dt = 0.1
sweep_times = 100
target_var = -100

def init_MPS() -> List[Tensor]:
    MPS: List[Tensor] = []

    init_vurtial_dim = [min(d**(i+1), D, d**(N-i-1)) for i in range(N-1)]
    # init_vurtial_dim = [D] * (N-1)

    bonds = ['s0', 'a1']
    value = random_unitary(d, init_vurtial_dim[0])
    tensor = Tensor(bonds, value)
    MPS.append(tensor)

    for i in range(1, N-1):
        value = random_unitary(init_vurtial_dim[i-1] * d, init_vurtial_dim[i])
        value = value.reshape(init_vurtial_dim[i-1], d, init_vurtial_dim[i])
        bonds = [f'a{i}', f's{i}', f'a{i+1}']
        tensor = Tensor(bonds, value)
        MPS.append(tensor)

    bonds = [f'a{N-1}', f's{N-1}']
    value = random_unitary(init_vurtial_dim[N-2], d)
    tensor = Tensor(bonds, value)
    MPS.append(tensor)
    return MPS

from matrix_gen import I, O, S_m, S_p, Z
# J = 2
# J_z = 1
# W = np.array([
#         I  , S_p, S_m, Z, O        ,
#         O  , O  , O  , O, J/2 * S_m,
#         O  , O  , O  , O, J/2 * S_p,
#         O  , O  , O  , O, J_z * Z  ,
#         O  , O  , O  , O, I
#     ]).reshape(5, 5, d, d)
# w0 = np.tensordot(np.array([1, 0, 0, 0, 0]), W, axes=([0], [0]))
# wN = np.tensordot(W, np.array([0, 0, 0, 0, 1]), axes=([1], [0])) # N-1
g = 1
J = 0
W = np.array([
    [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
    [1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0],
    [0, g, g, 0], [J, 0, 0, -J], [1, 0, 0, 1]
]).reshape(3, 3, 2, 2)
w0 = np.tensordot(np.array([0, 0, 1]), W, axes=([0], [0]))
wN = np.tensordot(W, np.array([1, 0, 0]), axes=([1], [0])) # N-1
def init_MPO() -> List[Tensor]:
    MPO: List[Tensor] = []
    MPO.append(Tensor(['b1', 's0', 's0p'], w0))
    for i in range(1, N-1):
        MPO.append(Tensor([f'b{i}', f'b{i+1}', f's{i}', f's{i}p'], W))
    MPO.append(Tensor([f'b{N-1}', f's{N-1}', f's{N-1}p'], wN))
    return MPO
def init_MPOsq() -> List[Tensor]:
    MPO_SQ: List[Tensor] = []
    temp1 = Tensor(['b1', 's0', '_a'], w0)
    temp2 = Tensor(['b1p', '_a', 's0p'], w0)
    MPO_SQ.append(temp1 * temp2)
    for i in range(1, N-1):
        temp1 = Tensor([f'b{i}', f'b{i+1}', f's{i}', f'_a'], W)
        temp2 = Tensor([f'b{i}p', f'b{i+1}p', f'_a', f's{i}p'], W)
        MPO_SQ.append(temp1 * temp2)
    temp1 = Tensor([f'b{N-1}', f's{N-1}', '_a'], wN)
    temp2 = Tensor([f'b{N-1}p', '_a', f's{N-1}p'], wN)
    MPO_SQ.append(temp1 * temp2)
    return MPO_SQ
def init_LR(MPS: List[Tensor], MPO: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
    MPS_DAG = [t.conj() for t in MPS]

    L_stack: List[Tensor] = []
    L_init_cursor = MPS[0]
    L_init_cursor = L_init_cursor * MPO[0]
    L_init_cursor = L_init_cursor * MPS_DAG[0]
    L_stack.append(L_init_cursor)
    for i in range(1, N-1):
        L_init_cursor = L_init_cursor * MPS[i]
        L_init_cursor = L_init_cursor * MPO[i]
        L_init_cursor = L_init_cursor * MPS_DAG[i]
        L_stack.append(L_init_cursor)
    return L_stack, []

def main():
    # init MPS
    MPS: List[Tensor] = init_MPS()

    # init MPO
    MPO: List[Tensor] = init_MPO()

    # init L, R
    L_stack, R_stack = init_LR(MPS, MPO)

    sweep_cursor = N - 1
    go_left = True
    
    energy_list = Recorder(xlabel="step", ylabel="energy")
    energy_list_site = Recorder(xlabel="site", ylabel="energy")

    for i in track(range(2 * sweep_times * (N - 1)), description="Sweeping..."):
        if go_left == True:
            if sweep_cursor == N - 1:
                H =  L_stack[-1] * MPO[sweep_cursor]
                expH = (-1j * dt * H).expm([b for b in H.bonds if not b.endswith('p')])
            else:
                H =  L_stack[-1] * MPO[sweep_cursor] * R_stack[-1]
                expH = (-1j * dt/2 * H).expm([b for b in H.bonds if not b.endswith('p')])

            update_site = MPS[sweep_cursor] * expH
            update_site.p()
            MPS[sweep_cursor] = update_site

            u, s, vh = MPS[sweep_cursor].svd(combine_bonds=[f'a{sweep_cursor}'],
                                        type="first",
                                        vh_new_bond_name='_Vh')
            MPS[sweep_cursor] = vh
            if sweep_cursor == N - 1:
                R_stack.append(MPS[sweep_cursor] * MPO[sweep_cursor] * MPS[sweep_cursor].conj())
            else:
                R_stack.append(R_stack[-1] * MPS[sweep_cursor] * MPO[sweep_cursor] * MPS[sweep_cursor].conj())
            K = L_stack[-1] * R_stack[-1]
            if sweep_cursor == N - 1:
                expK = (1j * dt * K).expm([b for b in K.bonds if not b.endswith('p')])
            else:
                expK = (1j * dt/2 * K).expm([b for b in K.bonds if not b.endswith('p')])
            
            update_site = (u * s) * expK
            update_site.p()
            MPS[sweep_cursor-1] = MPS[sweep_cursor-1] * update_site

            MPS[sweep_cursor-1].rename_bonds({'_Vh': f'a{sweep_cursor}'})
            MPS[sweep_cursor].rename_bonds({'_Vh': f'a{sweep_cursor}'})
            R_stack[-1].rename_bonds({'_Vh': f'a{sweep_cursor}', '_Vhp': f'a{sweep_cursor}p'})

            L_stack.pop()
            sweep_cursor -= 1
            if sweep_cursor == 0:
                go_left = False

            if sweep_cursor == 0:
                energy = (MPS[sweep_cursor] * MPO[sweep_cursor] * MPS[sweep_cursor].conj() * R_stack[-1]).value / N
            else:
                energy = (L_stack[-1] * MPS[sweep_cursor] * MPO[sweep_cursor] * MPS[sweep_cursor].conj() * R_stack[-1]).value / N
            energy_list.rec(energy)
            energy_list_site.rec(energy, sweep_cursor)


        if go_left == False:
            if sweep_cursor == 0:
                H =  R_stack[-1] * MPO[sweep_cursor]
                expH = (-1j * dt * H).expm([b for b in H.bonds if not b.endswith('p')])
            else:
                H =  L_stack[-1] * MPO[sweep_cursor] * R_stack[-1]
                expH = (-1j * dt/2 * H).expm([b for b in H.bonds if not b.endswith('p')])

            update_site = MPS[sweep_cursor] * expH
            update_site.p()
            MPS[sweep_cursor] = update_site

            u, s, vh = MPS[sweep_cursor].svd(combine_bonds=[f'a{sweep_cursor+1}'],
                                        type="second",
                                        u_new_bond_name='_U')
            MPS[sweep_cursor] = u
            if sweep_cursor == 0:
                L_stack.append(MPS[sweep_cursor] * MPO[sweep_cursor] * MPS[sweep_cursor].conj())
            else:
                L_stack.append(L_stack[-1] * MPS[sweep_cursor] * MPO[sweep_cursor] * MPS[sweep_cursor].conj())
            K = L_stack[-1] * R_stack[-1]
            if sweep_cursor == 0:
                expK = (1j * dt * K).expm([b for b in K.bonds if not b.endswith('p')])
            else:
                expK = (1j * dt/2 * K).expm([b for b in K.bonds if not b.endswith('p')])
            
            update_site = (s * vh) * expK
            update_site.p()
            MPS[sweep_cursor+1] = MPS[sweep_cursor+1] * update_site

            MPS[sweep_cursor+1].rename_bonds({'_U': f'a{sweep_cursor+1}'})
            MPS[sweep_cursor].rename_bonds({'_U': f'a{sweep_cursor+1}'})
            L_stack[-1].rename_bonds({'_U': f'a{sweep_cursor+1}', '_Up': f'a{sweep_cursor+1}p'})

            R_stack.pop()
            sweep_cursor += 1
            if sweep_cursor == N - 1:
                go_left = True

            if sweep_cursor == N - 1:
                energy = (MPS[sweep_cursor] * MPO[sweep_cursor] * MPS[sweep_cursor].conj() * L_stack[-1]).value / N
            else:
                energy = (L_stack[-1] * MPS[sweep_cursor] * MPO[sweep_cursor] * MPS[sweep_cursor].conj() * R_stack[-1]).value / N
            energy_list.rec(energy)
            energy_list_site.rec(energy, sweep_cursor)

    energy_list.plot()
    energy_list_site.plot()





if __name__ == "__main__":
    main()