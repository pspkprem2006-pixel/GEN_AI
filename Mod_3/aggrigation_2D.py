import numpy as np
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.sum())         # -> 21   (everything)
print(b.sum(axis=0))   # -> [5 7 9]   (sum down each COLUMN)
print(b.sum(axis=1))