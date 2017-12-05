

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import scipy




Na = 2000
Nb = 1000


mean = [-1, 4]
cov = [[10, -3], [-3, 3]] 

mean2 = [5, 1]
cov2 = [[-5, 2], [2, 6]]
x, x1 = np.random.multivariate_normal(mean, cov, Na).T
x2, x3 = np.random.multivariate_normal(mean2, cov2, Nb).T

fig = plt.figure()
plt.plot(x, x1, 'x' ,c='springgreen', alpha=.05/(0.9)**(2), zorder=0)
plt.plot(x2, x3, 'o',c='orange', alpha=.05/(0.9)**(2), zorder= 1)
plt.axis('equal')
plt.title("mean_ax: {} mean_ay: {} covA: {}  \n mean_bx: {} mean_by: {} covB: {} ".format(mean[0], mean[1], cov, mean2[0], mean2[1], cov2) )


print ('max x', max(x1))
X_1_grid = np.linspace(min(x),max(x),1000)
Y_1_grid = np.linspace(min(x1),max(x1),1000)

Y_1_cont, X_1_cont = np.meshgrid(X_1_grid, Y_1_grid)
Contour1 = mlab.bivariate_normal(Y_1_cont, X_1_cont, sigmax=cov[0][0], sigmay=cov[1][1], mux=mean[0], muy=mean[1], sigmaxy=cov[0][1])

Z_1_cont = Contour1**2

CS = plt.contour(Y_1_cont, X_1_cont, Z_1_cont)


X_2_grid = np.linspace(min(x2),max(x2),1000)
Y_2_grid = np.linspace(min(x3),max(x3),1000)

Y_2_cont, X_2_cont = np.meshgrid(X_2_grid, Y_2_grid)
Contour2 = mlab.bivariate_normal(Y_2_cont, X_2_cont, sigmax=cov2[0][0], sigmay=cov2[1][1], mux=mean2[0], muy=mean2[1], sigmaxy=cov2[0][1])
Z_2_cont = Contour2**2
CS = plt.contour(Y_2_cont, X_2_cont, Z_2_cont)



W = np.array([3, 4]).T

plt.arrow(mean[0], mean[1], W[0], W[1], head_width=0.5, head_length=0.5, fc='k', ec='k', zorder= 100)


X_1 = np.array([x, x1])
X_2 = np.array([x2, x3])



Y_a = np.array([ np.dot(W.T, np.array([X_1[0,n], X_1[1,n] ]))  for n in range(Na)])
print (Y_a[0])



Y_b = np.array([np.dot(W.T, np.array([X_2[0,n], X_2[1,n] ])) for n in range(Nb)])

print (Y_b.size)





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
plt.title("Both Y_a and Y_b not optimised")
plt.xlabel('Distribution')
plt.ylabel('Frequency')




ua = (1/Na)*sum(Y_a)
print (ua)

SigmaA_squared = sum((np.array([(Y_a[n]-ua)**2 for n in range(Na)])*(1/Na)))
print (SigmaA_squared)

ub = (1/Nb)*sum(Y_b)
print (ub)

SigmaB_squared = sum((np.array([(Y_b[n]-ub)**2 for n in range(Nb)])*(1/Nb)))
print (SigmaB_squared)

mean_difference = (ua - ub)**2

variance_difference = (Na/(Na+Nb))*SigmaA_squared + (Nb/(Na+Nb))*SigmaB_squared

print(mean_difference, variance_difference)

F_ratio = mean_difference/variance_difference

print (F_ratio)




plt.show()




















