import numpy as np

A = np.array([[1, 2],
              [-1, 1]])

B = np.array([[2, 0],
              [0, 2]])

C = np.array([[2, 0, -3],
              [0, 0, -1]])

D = np.array([[1, 2],
              [2, 3],
              [-1, 0]])

x = np.array([[1],
              [0]])

y = np.array([[0],
              [1]])

z = np.array([[1],
              [2],
              [-1]])

print("(a) A + B =")
print(A + B)
print()

print("(b) 3x - 4y =")
print(3 * x - 4 * y)
print()

print("(c) Ax =")
print(np.dot(A, x))
print()

print("(d) B(x - y) =")
print(np.dot(B, (x - y)))
print()

print("(e) Dx =")
print(np.dot(D, x))
print()

print("(f) Dy + z =")
print(np.dot(D, y) + z)
print()

print("(g) AB =")
print(np.dot(A, B))
print()

print("(h) BC =")
print(np.dot(B, C))
print()

print("(i) CD =")
print(np.dot(C, D))
print()

"""
Solution:
(a) A + B =
[[ 3  2]
 [-1  3]]

(b) 3x - 4y =
[[ 3]
 [-4]]

(c) Ax =
[[ 1]
 [-1]]

(d) B(x - y) =
[[ 2]
 [-2]]

(e) Dx =
[[ 1]
 [ 2]
 [-1]]

(f) Dy + z =
[[ 3]
 [ 5]
 [-1]]

(g) AB =
[[ 2  4]
 [-2  2]]

(h) BC =
[[ 4  0 -6]
 [ 0  0 -2]]

(i) CD =
[[5 4]
 [1 0]]
 "