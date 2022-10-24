import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))


def cost_1(z):
    return - math.log(sigmoid(z))


def cost_0(z):
    return - np.log(1-sigmoid(z))
z = np.arange(10,-10,0.1)
phi_z = sigmoid(z)
c1 =[]
c0 = []
for i in range(0,len(z)):
    c1.append(cost_1(z[i]))
    c0.append(cost_1(z[i]))


plt.plot(phi_z, c1,label='J(w) if y = 1', linestyle='--',color='b')
plt.plot(phi_z, c0, label='J(w) if y = 0', linestyle='-.',color='r')
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi(z)$')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()    