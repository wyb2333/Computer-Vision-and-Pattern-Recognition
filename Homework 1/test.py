import numpy as np
from conv import *
from Gauss_gen import *


a = np.zeros((3, 1))
for i in range(3):
    a[i, 0] = Gauss_fun(1.5, i - 1)
b = a.T
c = conv(a, b, mode_conv="full")
print(c / np.sum(c) == Gauss_gen(1.5, 3))

