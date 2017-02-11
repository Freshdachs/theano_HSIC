import numpy as np

def np_HSIC(x,y,k,l):
    m=len(x)
    H=np.eye(m)-1/m
    K=np_km(k,x)
    L=np_km(l,y)
    return np.trace(K @ H @ L @ H)*(m-1)**-2

def np_gk(sigma):
    return lambda x,y:np.exp(-sigma**(-2)*(x-y)**2)


def np_km(kernel,x):
    return np.array([[kernel(x_i,x_j) for x_j in x] for x_i in x])


def np_l(u):
    return np.median([np.abs(u[i]-u[j]) for i in range(len(u)) for j in range(i+1,len(u)) if u[i]-u[j]] )

def np_auto_HSIC(x,y):
    return np_HSIC(x, y, np_gk(np_l(x)), np_gk(np_l(y)))