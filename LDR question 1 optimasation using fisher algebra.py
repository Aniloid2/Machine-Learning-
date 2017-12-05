

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import scipy




Na = 2000
Nb = 1000


mean = [-1, 4]
cov = [[1, -5], [2, 10]]  

mean2 = [5, 1]
cov2 = [[5, 10], [2, 5]]
x, x1 = np.random.multivariate_normal(mean, cov, Na).T
x2, x3 = np.random.multivariate_normal(mean2, cov2, Nb).T



W = np.array([3, 4]).T




X_1 = np.array([x, x1])
X_2 = np.array([x2, x3])


Y_a = np.array([ np.dot(W.T, np.array([X_1[0,n], X_1[1,n] ]))  for n in range(Na)])
print (Y_a[0])



Y_b = np.array([np.dot(W.T, np.array([X_2[0,n], X_2[1,n] ])) for n in range(Nb)])

print (Y_b.size)


plt.show()


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


# - -----------------------------siris f ratio ------------------------------------
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


# - ----------------------------------------------------------------------------------

m1, m2 = np.array([1/Na*sum(x),1/Na*sum(x1)])   ,  np.array([1/Nb*sum(x2),1/Nb*sum(x3)])  


overall_mean = np.array([1/len(m1)*(m1[0]+m2[0]), 1/len(m1)*(m1[1]+m2[1])])


print ('overall mean',overall_mean)

W_op = np.array([3, 1]).T

print (W_op, m1, m2)

mk = np.dot(W_op,(m1-m2).T)

print (mk)



SW = np.zeros((2,2))

S1 = sum([np.dot(np.array([X_1[0,n], X_1[1,n]]).reshape(2,1) - m1.reshape(2,1) , (np.array([X_1[0,n], X_1[1,n]]).reshape(2,1) - m1.reshape(2,1)).T) for n in range(len(X_1[0])) ])

print (S1)
S2 = sum([ np.dot(np.array([X_2[0,n], X_2[1,n]]).reshape(2,1) - m2.reshape(2,1) , (np.array([X_2[0,n], X_2[1,n]]).reshape(2,1) - m2.reshape(2,1)).T) for n in range(len(X_2[0])) ])

SW = S1 +S2
print ('S1',S1, 'S2', S2)
print ('SW', SW) 

SB = np.zeros((2,2))



SBm1 = np.dot((m1.reshape(2,1) - overall_mean.reshape(2,1)), (m1.reshape(2,1) - overall_mean.reshape(2,1)).T)
print ('sbm1', SBm1)
SB += SBm1
SBm2 = np.dot((m2.reshape(2,1) - overall_mean.reshape(2,1)), (m2.reshape(2,1) - overall_mean.reshape(2,1)).T)

SB += SBm2

print ('SB',SB)

eig_val, eig_vec = np.linalg.eig(np.linalg.inv(SW).dot(SB))


for i in range(len(eig_val)):
	eig_vector_row = eig_vec[:,i].reshape(2,1)
	eig_value = eig_val[i]
	print ('eignvector', i, '\n', eig_vector_row, '\n eignvalue' , i ,'\n', '{:.2e}'.format(eig_value))



eig_pairs = [ (np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val)) ]

eig_pairs = sorted(eig_pairs , key=lambda k: k[0], reverse=True )

for i in eig_pairs:
	print (i[0], i[1])


W_op = eig_pairs[0][1]

print (W_op)

X_lda_a = np.dot(X_1.T, W_op)
X_lda_b = np.dot(X_2.T, W_op)

fig = plt.figure()


plt.hist(X_lda_a, color= ['springgreen'], bins="auto" ,edgecolor='None', alpha = 0.5)
plt.hist(X_lda_b, color= [ 'orange'] ,bins="auto" ,edgecolor='None', alpha = 0.5)
plt.title("Both Y_a and Y_b optimised")
plt.xlabel('Distribution')
plt.ylabel('Frequency')




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




plt.arrow(overall_mean[0], overall_mean[1], -10*W_op[0], -10*W_op[1], head_width=0, head_length=0, fc='b', ec='b', zorder= 100)


plt.arrow(overall_mean[0], overall_mean[1], 10*W_op[0], 10*W_op[1], head_width=0, head_length=0, fc='b', ec='b', zorder= 100)

plt.arrow(overall_mean[0], overall_mean[1], -10*W[0], -10*W[1], head_width=0, head_length=0, fc='r', ec='r', zorder= 100)


plt.arrow(overall_mean[0], overall_mean[1], 10*W[0], 10*W[1], head_width=0, head_length=0, fc='r', ec='r', zorder= 100)



plt.show()