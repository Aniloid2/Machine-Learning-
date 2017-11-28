
# K number of classes 

# Ck class number from (1.2.4...k)

# x input vector 

# p(ck|x) probability that ck belongs to x

# y linear prediction 

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import scipy


# mua, mub = 3, 4 

# sigmaa, sigmab = 10, 30

# Na = np.random.normal(mua, sigmaa, 1000)
# Nb = np.random.normal(mub, sigmab, 1000)

# print (Na, Nb)


# count, bins, ignored = plt.hist(Na, 30, normed=True)
# plt.plot(bins, 1/(sigmaa * np.sqrt(2 * np.pi)) * np.exp( - (bins - mua)**2 / (2 * sigmaa**2) ), linewidth=2, color='r')
# count, bins, ignored = plt.hist(Nb, 30, normed=True)
# plt.plot(bins, 1/(sigmab * np.sqrt(2 * np.pi)) * np.exp( - (bins - mub)**2 / (2 * sigmab**2) ), linewidth=2, color='r')


# plt.show()

Na = 2000
Nb = 1000


mean = [-1, 4]
cov = [[1, -5], [2, 10]]  # diagonal covariance

mean2 = [5, 1]
cov2 = [[5, 10], [2, 5]]
x, x1 = np.random.multivariate_normal(mean, cov, Na).T
x2, x3 = np.random.multivariate_normal(mean2, cov2, Nb).T
# print (x,y)
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

# print (x.size)

# print (x[0], y[0])

X_1 = np.array([x, x1])
X_2 = np.array([x2, x3])

# print (X_1[0,1], X_2[0])

# print (len(X))

# Y = [np.dot(W,Xn) for Xn in x]
# Y_a = np.array([[ W[0]*X_1[0,n],W[1]*X_1[1,n]] for n in range(Na)])

Y_a = np.array([ np.dot(W.T, np.array([X_1[0,n], X_1[1,n] ]))  for n in range(Na)])
print (Y_a[0])

# Y_a = np.array([ W*np.array([X_1[0,n],X_1[1,n] ]) for n in range(Na)])
# print (X[0])
# Y = []
# for n in range(x.size):
# 	Yx = W[0]*X[0,n]
# 	Yy = W[1]*X[1,n]
# 	Y.append(np.array([Yx, Yy]))
# Y_b = np.array([[ W[0]*X_1[0,n],W[1]*X_1[1,n]] for n in range(Nb)])

Y_b = np.array([np.dot(W.T, np.array([X_2[0,n], X_2[1,n] ])) for n in range(Nb)])

print (Y_b.size)




# print (Y_a[0,0], Y_a[0,1])

# Yx, Yy = Y_a[:,0], Y_a[:,1]

# print (Yx[0], Yy[0])


fig = plt.figure()
# print(bins)

# plt.hist(Y_a, color= ['springgreen', 'green'], bins="auto")
plt.hist(Y_a,color= ['springgreen'], bins="auto")
plt.title("Y_a dot product W x_1, Na: {}".format(Na))
plt.xlabel('Distribution')
plt.ylabel('Frequency')

fig = plt.figure()
# plt.hist(Y_b, color= ['yellow', 'orange'], bins="auto")
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
# plt.show()



ua = (1/Na)*sum(Y_a)
print (ua)

SigmaA_squared = sum((np.array([(Y_a[n]-ua)**2 for n in range(Na)])*(1/Na)))
print (SigmaA_squared)

ub = (1/Nb)*sum(Y_b)
print (ub)

SigmaB_squared = sum((np.array([(Y_b[n]-ub)**2 for n in range(Nb)])*(1/Nb)))
print (SigmaB_squared)

mean_difference = (ua - ub)**2 # has to be positive

variance_difference = (Na/(Na+Nb))*SigmaA_squared + (Nb/(Na+Nb))*SigmaB_squared

print(mean_difference, variance_difference)

F_ratio = mean_difference/variance_difference

print (F_ratio)

# plt.figure()

# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# print ('x:', x, y )
# X, Y = np.meshgrid(x, y)
# Contour1 = mlab.bivariate_normal(X, Y, sigmax=cov[0][0], sigmay=cov[1][1], mux=mean[0], muy=mean[1], sigmaxy=cov[0][1])
# Contour2 = mlab.bivariate_normal(X, Y, sigmax=cov2[0][0], sigmay=cov2[1][1], mux=mean2[0], muy=mean2[1], sigmaxy=cov2[0][1])


# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)

# plt.contour(Contour)


# Z = 10.0 * (Contour2 - Contour1)
# CS = plt.contour(Contour1, Contour2, Z)


plt.show()


# hey man, i thought you might know this, with multivariate_normal

 # we get a x and y value, but when calculating y = w*x do we just ignore y the y values?
























# N = 200
# # prob = [1/3]*3
# num_throws = 2000
# prob = [0.25, 0.25, 0.5]

# prob2 = [0.24, 0.22,0.55 ]

# print (prob)

# X = np.random.multinomial(num_throws, prob, size=N)

# Y = np.random.multinomial(num_throws, prob2, size=N)


# [n1,n2] = [X[:,0], X[:,1]]
# [n3, n4]= [Y[:,0], Y[:,1]]

# # print ('Sample_vectorised on Y:',n1, n2 ,n3)

# [meanN1, meanN2] = [np.mean(n1), np.mean(n2)]
# [meann3, meann4] = [np.mean(n3), np.mean(n4)]

# print ('Means: \n',meanN1, meanN2)



# E11 =(1/N-1)*np.dot((n1-meanN1),(n1-meanN1))
# E12 = (1/N-1)*np.dot((n1-meanN1),(n2-meanN2))
# E21 = (1/N-1)*sum(((n2-meanN2)*(n1- meanN1)))
# E22 = (1/N-1)*sum(((n2-meanN2)*(n2-meanN2)))




# E = np.matrix([[E11, E12], [E21, E22] ])



# print ('Covariance matrix: \n',E)



# y = wx



# fig  = plt.scatter(n1,n2, c='springgreen',marker='.', alpha=.05/(0.5)**(2))

# plt.scatter(n3,n4, c='blue',marker='.', alpha=.05/(0.5)**(2))

# plt.title('The probability is :'+ str(prob))

# plt.scatter(meanN1,meanN2, s=10, c='r')
# # plt.axis([0,6000,0,6000])

# plt.show()
