
# import numpy as np
# a=np.array([[1,1,1],[2,3,4]])
# b=np.array([[1,2,1],[2,3,4]])
# c=[]

# for i in range(2):
	# sum=0
	# for j in range(3):
		# sum+=(a[i][j]-b[i][j])**2
	# c.append(((sum)**1/2)/14.9)
# print(c)

import numpy as np
n_epochs=100
for epoch in np.arange(1, n_epochs + 1):
	print(epoch)