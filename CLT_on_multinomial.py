import matplotlib.pyplot as plt
import numpy as np
import scipy





N = 20000
num_throws = 20000
prob = [1/3, 1/3, 1/3 ]
print (prob)

X = np.random.multinomial(num_throws, prob, size=N)

[n1,n2,n3] = [X[:,0], X[:,1], X[:,2]]


[meanN1, meanN2, meanN3] = [np.mean(n1), np.mean(n2), np.mean(n3)]



E11 =(1/N-1)*np.dot((n1-meanN1),(n1-meanN1))
E12 = (1/N-1)*np.dot((n1-meanN1),(n2-meanN2))
E21 = (1/N-1)*sum(((n2-meanN2)*(n1- meanN1)))
E22 = (1/N-1)*sum(((n2-meanN2)*(n2-meanN2)))




E = np.matrix([[E11, E12], [E21, E22] ])


print ('Covariance matrix: \n',E)



fig  = plt.scatter(n1,n2, c='springgreen',marker='.', alpha=.05/(0.5)**(2))

plt.title('The probability is :'+ str(prob))

plt.scatter(meanN1,meanN2, s=10, c='r')


plt.show()

