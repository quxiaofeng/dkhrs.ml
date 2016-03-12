import numpy as np
from scipy import signal

A = np.array([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])
B = np.array([[1, 1], [1, 1]])

signal.convolve2d(A, B)
# array([[ 1,  3,  5,  4,  1],
#        [ 5, 12, 16, 11,  2],
#        [11, 24, 28, 17,  2],
#        [ 7, 15, 17, 10,  1]])

 signal.convolve2d(A, B, mode='valid')
 # array([[12, 16, 11],
 #       [24, 28, 17]])
