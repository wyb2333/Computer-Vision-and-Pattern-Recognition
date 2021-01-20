import numpy as np


a = range(27)
a = np.array(a)
a = a.reshape([3, 3, 3])
b = range(9)
b = np.array(b)
b = b.reshape([1, 3, 3])
print(a)
# print(a[0, :, :])
# print(np.sum(a, 0))
print(b)
print(a*b)
