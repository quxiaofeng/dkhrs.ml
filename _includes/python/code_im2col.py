import numpy as np

def im2col(A, B):
	# Parameters
	M,N = A.shape
	col_extent = N - B[1] + 1
	row_extent = M - B[0] + 1

	# Get Starting block indices
	start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])

	# Get offsetted indices across the height and width of input array
	offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

	# Get all actual indices & index into input array for final output
	out = np.take (A,start_idx.ravel()[:,None] + offset_idx.ravel())

	return out

A = np.array([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])

im2col(A, [2, 2])
# array([[1, 2, 3, 4, 5, 6],
#        [2, 3, 1, 5, 6, 1],
#        [4, 5, 6, 7, 8, 9],
#        [5, 6, 1, 8, 9, 1]])
