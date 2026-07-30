import numpy as np
data = np.array([15, 22, 8, 19, 31, 12])
print(data > 20)          # -> [False  True False False  True False]
print(data[data > 20])