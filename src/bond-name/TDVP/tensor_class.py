from typing import List, Tuple, Final, Literal
from scipy.sparse.linalg import eigsh, expm, LinearOperator, expm_multiply
import numpy as np

def partial_square_ratio(arr, n):
    """
    計算後 n 項平方和佔整個陣列平方和的比例
    參數:
        arr: ndarray 或 list
        n: int 後 n 項
    回傳:
        float 值介於 [0, 1]
    """
    arr = np.asarray(arr)
    if n < 0:
        raise ValueError("n 必須是非負整數")
    elif n == 0:
        return 0.0
    total = np.sum(arr**2)
    if total == 0:
        return 0.0
    partial = np.sum(arr[n:]**2)
    return partial / total

class Tensor:
    def __init__(self, bonds: list[str], value: np.ndarray):
        if len(bonds) != value.ndim:
            raise ValueError("Number of bonds must match the number of dimensions in value")
        self.bonds = bonds
        self.value = value

    def bond_dim(self, bond_name: str) -> int:
        """Return the dimension (size) of a given bond (axis)."""
        try:
            idx = self.bonds.index(bond_name)
            return self.value.shape[idx]
        except ValueError:
            raise KeyError(f"Bond '{bond_name}' not found in bonds {self.bonds}")
    def rename_bonds(self, rename_map: dict[str, str]) -> "Tensor":
        if [b for b in rename_map.keys() if b not in self.bonds] != []:  # All bond are include in self.bonds
            raise ValueError("Can't find the bond.")
        self.bonds = [rename_map.get(b, b) for b in self.bonds]
        return self
    
    def copy(self) -> "Tensor":
        return self
    
    def conj(self) -> "Tensor":  # Warning!!!: this is for specific use since I'm tired
        return Tensor([
                        b if b.startswith('b')
                        else b[:-1] if b.endswith('p')
                        else b + 'p'
                        for b in self.bonds
                      ]
                      , self.value.conj())
    def p(self, inplace: bool = True) -> "Tensor":  # Warning!!!: this is for specific use since I'm tired
        if inplace:
            self.bonds = [
                        b if b.startswith('b')
                        else b[:-1] if b.endswith('p')
                        else b + 'p'
                        for b in self.bonds
                      ]
            return self
        return Tensor([
                        b if b.startswith('b')
                        else b[:-1] if b.endswith('p')
                        else b + 'p'
                        for b in self.bonds
                      ]
                      , self.value)
    def expm(self, combine_bonds: list[str]) -> "Tensor":
        if [b for b in combine_bonds if b not in self.bonds] != []:  # All bond are include in self.bonds
            raise ValueError("Can't find the bond.")
        elif len(combine_bonds) == 0 or len(combine_bonds) == len(self.bonds):
            raise ValueError("Can't become a matrix.")
        
        other_bonds = [b for b in self.bonds if b not in combine_bonds]

        # Step 2: 將 find_bonds 移到前面 (transpose)
        perm = [self.bonds.index(b) for b in combine_bonds + other_bonds]
        transposed = self.value.transpose(perm)

        # Step 3: reshape 成矩陣
        shape = transposed.shape
        dim_row = int(np.prod(shape[:len(combine_bonds)]))
        dim_col = int(np.prod(shape[len(combine_bonds):]))
        matrix = transposed.reshape(dim_row, dim_col)

        # Step 4: 求 eigenvalues/vectors
        exp_mat = expm(matrix)

        # Step 5: reshape 成 Tensor（find_bond 的 shape）
        exp_ten = exp_mat.reshape(shape)
        
        return Tensor(combine_bonds + other_bonds, exp_ten)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return self.contract(other)
        elif isinstance(other, (int, float, complex, np.number)):
            return Tensor(self.bonds, self.value * other)
        else:
            return NotImplemented
    def __rmul__(self, other):
        return self.__mul__(other)
    def contract(self, other: "Tensor") -> "Tensor":
        # 找出要做 contraction 的腳（同名腳）
        common_bonds = [b for b in self.bonds if b in other.bonds]
        
        # 自己與對方的軸索引
        self_axes = [self.bonds.index(b) for b in common_bonds]
        other_axes = [other.bonds.index(b) for b in common_bonds]
        
        # 用 np.tensordot 做 contraction
        result_value = np.tensordot(self.value, other.value, axes=(self_axes, other_axes))

        # 建立新 bonds：保留未被 contract 的腳
        self_remain = [b for b in self.bonds if b not in common_bonds]
        other_remain = [b for b in other.bonds if b not in common_bonds]
        result_bonds = self_remain + other_remain

        return Tensor(result_bonds, result_value)
    
    def ground_state(self, find_bonds: list[str]) -> "Tensor":
        if [b for b in find_bonds if b not in self.bonds] != []:  # All bond are include in self.bonds
            raise ValueError("Can't find the bond.")
        if len(find_bonds) == 0 or len(find_bonds) == len(self.bonds):
            raise ValueError("Can't become a matrix.")
        
        other_bonds = [b for b in self.bonds if b not in find_bonds]

        # Step 2: 將 find_bonds 移到前面 (transpose)
        perm = [self.bonds.index(b) for b in find_bonds + other_bonds]
        transposed = self.value.transpose(perm)

        # Step 3: reshape 成矩陣
        shape = transposed.shape
        dim_row = int(np.prod(shape[:len(find_bonds)]))
        dim_col = int(np.prod(shape[len(find_bonds):]))
        matrix = transposed.reshape(dim_row, dim_col)

        # Step 4: 檢查是否為方陣（若不是就不能做 eigen decomposition）
        if dim_row != dim_col:
            raise ValueError(f"Matrix must be square: got {dim_row} x {dim_col}")

        # Step 5: 求 eigenvalues/vectors
        _, eigvecs = eigsh(matrix, k=1, which='SA')
        ground_vec = eigvecs[:, 0]

        # Step 6: reshape 成 Tensor（find_bond 的 shape）
        tensor_shape = shape[:len(find_bonds)]
        ground_tensor = ground_vec.reshape(tensor_shape)

        return Tensor(find_bonds, ground_tensor)
    
    def svd(self,
            combine_bonds: list[str],
            type: Literal["first", "second"] = "first",
            u_new_bond_name: str = "_U",
            vh_new_bond_name: str = "_Vh",
            error: float = 0,
            dim_cutoff: int = None
        ) -> Tuple["Tensor", "Tensor", "Tensor"]:
        
        if [b for b in combine_bonds if b not in self.bonds] != []:  # All bond are include in self.bonds
            raise ValueError("Can't find the bond.")
        elif len(combine_bonds) == 0 or len(combine_bonds) == len(self.bonds):
            raise ValueError("Can't become a matrix.")
        
        if type == "first":
            first_bonds = combine_bonds
            secend_bonds = [b for b in self.bonds if b not in combine_bonds]
        else:
            secend_bonds = combine_bonds
            first_bonds = [b for b in self.bonds if b not in combine_bonds]

        # Step 2: 將 find_bonds 移到前面 (transpose)
        perm = [self.bonds.index(b) for b in first_bonds + secend_bonds]
        transposed = self.value.transpose(perm)

        # Step 3: reshape 成矩陣
        shape = transposed.shape
        dim_row = int(np.prod(shape[:len(first_bonds)]))
        dim_col = int(np.prod(shape[len(first_bonds):]))
        matrix = transposed.reshape(dim_row, dim_col)

        # Step 5: svd
        u, s, vh = np.linalg.svd(matrix)

        # Step 6: 找符合error的維數，並修剪s, v

        # Step 7: 打包成 Tensor
        s_dim = min(len(s), dim_cutoff or float('inf'))
        
        err = partial_square_ratio(s, len(s) - s_dim)
        # print(f"\033[91m err:{err} \033[00m")

        u = u[:, :s_dim]
        s = s[:s_dim]
        vh = vh[:s_dim, :]

        u_shape = shape[:len(first_bonds)] + (s_dim,)
        u = u.reshape(u_shape)
        u_tensor = Tensor(first_bonds + [u_new_bond_name], u)

        vh_shape = (s_dim,) + shape[len(first_bonds):]
        vh = vh.reshape(vh_shape)
        vh_tensor = Tensor([vh_new_bond_name] + secend_bonds, vh)

        s_tensor = Tensor([u_new_bond_name, vh_new_bond_name], np.diag(s))

        return u_tensor, s_tensor, vh_tensor

    def __repr__(self):
        return (f"Tensor(bonds={self.bonds}, shape={self.value.shape})")
    