import numpy as np
import matplotlib.pyplot as plt

# Generate data...
x = np.array([2,3,3,3.5,6,7])
y = np.array([8,3,3,3.5,6,7])
z= np.array([1,1,1,1,1,2])
# Plot...
plt.scatter(x, y, c=z, s=30)

plt.xlabel('ae1')
plt.ylabel('ae2')
plt.show()