

import matplotlib.pyplot as plt
import numpy as np
import scipy


Na = 2000
Nb = 1000


mean = [-30, 0]
cov = [[1, 40], [-250, 100]]

mean2 = [5, 1]
cov2 = [[5, 20], [40, 100]]
x, x1 = np.random.multivariate_normal(mean, cov, Na).T
x2, x3 = np.random.multivariate_normal(mean2, cov2, Nb).T

fig = plt.figure()
plt.plot(x, x1, 'x' ,c='springgreen', alpha=.05/(0.2)**(2))
plt.plot(x2, x3, 'o',c='orange', alpha=.05/(0.2)**(2))
plt.axis('equal')
plt.title("mean_ax: {} mean_ay: {} covA: {}  \n mean_bx: {} mean_by: {} covB: {} ".format(mean[0], mean[1], cov, mean2[0], mean2[1], cov2) )


X_1 = np.array([x, x1])
X_2 = np.array([x2, x3])

W = np.array([1, 9]).T
Y_a = np.array([ np.dot(W.T, np.array([X_1[0,n], X_1[1,n] ]))  for n in range(Na)])
Y_b = np.array([np.dot(W.T, np.array([X_2[0,n], X_2[1,n] ])) for n in range(Nb)])


fig = plt.figure()


plt.hist(Y_a, color= ['springgreen'], bins="auto" ,edgecolor='None', alpha = 0.5)
plt.hist(Y_b, color= [ 'orange'] ,bins="auto" ,edgecolor='None', alpha = 0.5)
plt.title("Both Y_a and Y_b NOT optimised")
plt.xlabel('Distribution')
plt.ylabel('Frequency')


best_W = [0 , 0]
best_F = 0
for i in range(30):
	for j in range(30):
		W = np.array([i, j]).T


		Y_a = np.array([ np.dot(W.T, np.array([X_1[0,n], X_1[1,n] ]))  for n in range(Na)])

		Y_b = np.array([np.dot(W.T, np.array([X_2[0,n], X_2[1,n] ])) for n in range(Nb)])

		ua = (1/Na)*sum(Y_a)


		SigmaA_squared = sum((np.array([(Y_a[n]-ua)**2 for n in range(Na)])*(1/Na)))


		ub = (1/Nb)*sum(Y_b)


		SigmaB_squared = sum((np.array([(Y_b[n]-ub)**2 for n in range(Nb)])*(1/Nb)))


		mean_difference = (ua - ub)**2

		variance_difference = (Na/(Na+Nb))*SigmaA_squared + (Nb/(Na+Nb))*SigmaB_squared



		F_ratio = mean_difference/variance_difference


		if F_ratio > best_F:
			print (F_ratio, best_W)
			best_F = F_ratio
			best_W = W







print (best_W, best_F)



Y_a = np.array([ np.dot(best_W.T, np.array([X_1[0,n], X_1[1,n] ]))  for n in range(Na)])
Y_b = np.array([np.dot(best_W.T, np.array([X_2[0,n], X_2[1,n] ])) for n in range(Nb)])


fig = plt.figure()

plt.hist(Y_a,color= ['springgreen'], bins="auto")
plt.title("Y_a dot product W x_1, Na: {}".format(Na))
plt.xlabel('Distribution')
plt.ylabel('Frequency')

fig = plt.figure()

plt.hist(Y_b, color= [ 'orange'] ,bins="auto")
plt.title("Y_b dot product W x_2, Nb: {}".format(Nb))
plt.xlabel('Distribution')
plt.ylabel('Frequency')

fig = plt.figure()


plt.hist(Y_a, color= ['springgreen'], bins="auto" ,edgecolor='None', alpha = 0.5)
plt.hist(Y_b, color= [ 'orange'] ,bins="auto" ,edgecolor='None', alpha = 0.5)
plt.title("Both Y_a and Y_b optimised")
plt.xlabel('Distribution')
plt.ylabel('Frequency')

plt.show()














