import numpy as np
import theano
import theano.tensor as T
from functools import *

def HSIC(x,y,k,l):
    m=x.shape[0]
    H=T.eye(m)-1./m
    K=kern_mat_t(k,x)
    L=kern_mat_t(l,y)
    return reduce(T.dot,[K , H , L , H]).trace() *(m-1.)**-2



def kern_mat(kernel,x):
    return np.array([[kernel(x_i,x_j) for x_j in x] for x_i in x])

def gaussian_kernel(sigma):
    return lambda x,y:np.exp(-sigma**(-2.)*(x-y)**2.)


# median heuristic
def l(u):
    return np.median([np.abs(u[i]-u[j]) for i in range(len(u)) for j in range(i+1,len(u)) if u[i]-u[j]] )

def l_t(u):
    return t_median( T.triu(kern_mat_t(lambda a, b: np.abs(a - b), u)).nonzero_values())

def t_median(tensor):
    return T.switch(T.eq( (tensor.shape[0] % 2), 0),
        #if even vector
        T.mean(T.sort(tensor)[ ((tensor.shape[0]//2)-1) : ((tensor.shape[0]//2)+1) ]),
        #if odd vector
        T.sort(tensor)[tensor.shape[0]//2])

def kern_mat_t(kernel,x):
    x_=x.dimshuffle(0,'x')
    return  kernel(x_,x)

x = T.dvector('x')
y = T.dvector('y')
s_x = T.dscalar('s_x')
s_y = T.dscalar('s_y')

gaussian_HSIC = theano.function([x,y,s_x,s_y],HSIC(x,y,gaussian_kernel(s_x),gaussian_kernel(s_y)))

auto_HSIC = theano.function([x,y],HSIC(x,y,gaussian_kernel(l_t(x)),gaussian_kernel(l_t(y))))


corr = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]],100)
uncorr = np.random.multivariate_normal([0,0],[[1,0],[0,1]],100)

a , b = corr[:,0], corr[:,1]
c , d = uncorr[:,0], uncorr[:,1]

t_x = {x:a}