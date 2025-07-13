from tensor_class import Tensor
import numpy as np

a = Tensor(['a1', 'a2'], np.array([[1,0],[0,2]]))
a = a.expm(['a2'])
print(a)
print(a.value)