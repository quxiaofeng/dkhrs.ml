A = [1 2; 3 4]
# 2x2 Array{Int64,2}:
#  1  2
#  3  4

@doc vec
# vec(Array) -> Vector
#
# Vectorize an array using column-major convention.

vec(A)
# 4-element Array{Int64,1}:
#  1
#  3
#  2
#  4
