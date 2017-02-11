import numpy as np

def HSIC(x,y,k,l):
    m=len(x)
    H=np.eye(m)-1/m
    K=km(k,x)
    L=km(l,y)
    return np.trace(K @ H @ L @ H)*(m-1)**-2

def gk(sigma):
    return lambda x,y:np.exp(-sigma**(-2)*(x-y)**2)


def km(kernel,x):
    return np.array([[kernel(x_i,x_j) for x_j in x] for x_i in x])


def l(u):
    return np.median([np.abs(u[i]-u[j]) for i in range(len(u)) for j in range(i+1,len(u)) if u[i]-u[j]] )

def auto_HSIC(x,y):
    return HSIC(x, y, gk(l(x)), gk(l(y)))