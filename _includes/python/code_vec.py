import numpy as np

A = np.array([[1, 2], [3, 4]])
A
# array([[1, 2],
#        [3, 4]])

A.flatten(1)
# array([1, 3, 2, 4])

def vec(A):
    return A.flatten(1)

vec(A)
# array([1, 3, 2, 4])
