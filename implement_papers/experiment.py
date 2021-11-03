import numpy as np

a = np.array([9,7,8,5])
b = np.arange(a.shape[0])

mask = [False, True, False, True]

idx = b[mask]

print("index list :",idx)
print("value :",a[idx])
