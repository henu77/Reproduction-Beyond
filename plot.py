import matplotlib.pyplot as plt

import numpy as np

data = np.loadtxt('model_replica/val_output.csv', delimiter=',', dtype=str)

print(data.shape)
data = data[:, :-1]
print(data.shape)
data = data.astype(np.float32)
print(data.shape)
plt.figure(figsize=(12, 8))
plt.plot(data[:, 2])
plt.show()
plt.close()
plt.figure(figsize=(12, 8))
plt.plot(data[:, 3])
plt.show()
plt.close()
plt.figure(figsize=(12, 8))
plt.plot(data[:, 4])
plt.show()
plt.close()
plt.figure(figsize=(12, 8))
plt.plot(data[:, 5])
plt.show()
plt.close()
plt.figure(figsize=(12, 8))
plt.plot(data[:, 6])
plt.show()
plt.close()
plt.figure(figsize=(12, 8))
plt.plot(data[:, 7])
plt.show()
plt.close()