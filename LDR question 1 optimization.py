

import matplotlib.pyplot as plt
import numpy as np
import scipy


Na = 2000
Nb = 1000


mean = [-30, 0]
cov = [[1, 40], [-250, 100]]  # diagonal covariance

mean2 = [5, 1]
cov2 = [[5, 20], [40, 100]]
x, x1 = np.random.multivariate_normal(mean, cov, Na).T
x2, x3 = np.random.multivariate_normal(mean2, cov2, Nb).T
# print (x,y)
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
		# print (ua)

		SigmaA_squared = sum((np.array([(Y_a[n]-ua)**2 for n in range(Na)])*(1/Na)))
		# print (SigmaA_squared)

		ub = (1/Nb)*sum(Y_b)
		# print (ub)

		SigmaB_squared = sum((np.array([(Y_b[n]-ub)**2 for n in range(Nb)])*(1/Nb)))
		# print (SigmaB_squared)

		mean_difference = (ua - ub)**2 # has to be positive

		variance_difference = (Na/(Na+Nb))*SigmaA_squared + (Nb/(Na+Nb))*SigmaB_squared

		# print(mean_difference, variance_difference)

		F_ratio = mean_difference/variance_difference

		# print (F_ratio)
		if F_ratio > best_F:
			print (F_ratio, best_W)
			best_F = F_ratio
			best_W = W







print (best_W, best_F)



Y_a = np.array([ np.dot(best_W.T, np.array([X_1[0,n], X_1[1,n] ]))  for n in range(Na)])
Y_b = np.array([np.dot(best_W.T, np.array([X_2[0,n], X_2[1,n] ])) for n in range(Nb)])


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
plt.title("Both Y_a and Y_b optimised")
plt.xlabel('Distribution')
plt.ylabel('Frequency')

plt.show()
















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
