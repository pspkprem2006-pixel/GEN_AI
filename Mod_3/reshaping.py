import numpy as np
a = np.arange(1, 13)          
b = a.reshape(3, 4)           
print(b)
print(b.reshape(-1)) 