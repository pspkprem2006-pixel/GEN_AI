import numpy as np
print(np.zeros(5))          # -> [0. 0. 0. 0. 0.]      (five zeros)
print(np.ones((2, 3)))      # -> a 2x3 array of ones
print(np.arange(0, 10, 2))  # -> [0 2 4 6 8]           (like range(), but an array)
print(np.linspace(0, 1, 5)) # -> [0.   0.25 0.5  0.75 1.  ]  (5 evenly spaced values)
print(np.random.default_rng(42).integers(1, 7, 5))  # -> 5 random dice rolls, e.g. [1 6 2 6 5]
