import numpy as np
import theano
import theano.tensor as T
from functools import reduce

def HSIC(x, y, k, l):
    m=x.shape[0]
    H=T.eye(m)-1./m
    K=km(k,x)
    L=km(l,y)
    return reduce(T.dot,[K , H , L , H]).trace() *(m-1.)**-2

def gk(sigma):
    return lambda x,y:np.exp(-sigma**(-2.)*(x-y)**2.)

def l(u):
    return t_median( T.triu(km(lambda a, b: np.abs(a - b), u)).nonzero_values())

def t_median(tensor):
    return T.switch(T.eq( (tensor.shape[0] % 2), 0),
        #if even vector
        T.mean(T.sort(tensor)[ ((tensor.shape[0]//2)-1) : ((tensor.shape[0]//2)+1) ]),
        #if odd vector
        T.sort(tensor)[tensor.shape[0]//2])

def km(kernel,x):
    x_=x.dimshuffle(0,'x')
    return  kernel(x_,x)


x = T.dvector('x')
y = T.dvector('y')
s_x = T.dscalar('s_x')
s_y = T.dscalar('s_y')

gaussian_HSIC = theano.function([x,y,s_x,s_y], HSIC(x, y, gk(s_x), gk(s_y)))

auto_HSIC = theano.function([x,y], HSIC(x, y, gk(l(x)), gk(l(y))))