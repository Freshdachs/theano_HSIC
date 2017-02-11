import numpy as np
import numpy as np
import theano
import theano.tensor as T
from functools import *

def t_HSIC(x, y, k, l):
    m=x.shape[0]
    H=T.eye(m)-1./m
    K=t_km(k,x)
    L=t_km(l,y)
    return reduce(T.dot,[K , H , L , H]).trace() *(m-1.)**-2

def t_gk(sigma):
    return lambda x,y:np.exp(-sigma**(-2.)*(x-y)**2.)

def t_l(u):
    return t_median( T.triu(t_km(lambda a, b: np.abs(a - b), u)).nonzero_values())

def t_median(tensor):
    return T.switch(T.eq( (tensor.shape[0] % 2), 0),
        #if even vector
        T.mean(T.sort(tensor)[ ((tensor.shape[0]//2)-1) : ((tensor.shape[0]//2)+1) ]),
        #if odd vector
        T.sort(tensor)[tensor.shape[0]//2])

def t_km(kernel,x):
    x_=x.dimshuffle(0,'x')
    return  kernel(x_,x)


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


x = T.dvector('x')
y = T.dvector('y')
s_x = T.dscalar('s_x')
s_y = T.dscalar('s_y')

corr = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]],100)
uncorr = np.random.multivariate_normal([0,0],[[1,0],[0,1]],100)

a , b = corr[:,0], corr[:,1]
c , d = uncorr[:,0], uncorr[:,1]

t_x = {x:a}

gaussian_HSIC = theano.function([x,y,s_x,s_y], t_HSIC(x, y, t_gk(s_x), t_gk(s_y)))

t_auto_HSIC = theano.function([x,y], t_HSIC(x, y, t_gk(t_l(x)), t_gk(t_l(y))))

def np_auto_HSIC(x,y):
    return np_HSIC(x, y, np_gk(np_l(x)), np_gk(np_l(y)))

def test_median():
    assert np.allclose(np_l(a),t_l(x).eval({x:a}))

def test_gaussian_kernel_same_sigma():
    sigma = np_l(a)
    t_sigma = T.dscalar("t_sigma")
    assert np.allclose(np_gk(sigma)(3,4),t_gk(t_sigma)(3,4).eval({t_sigma:sigma}))
    assert np.allclose(np_gk(sigma)(a[0], a[-1]), t_gk(t_sigma)(a[0], a[-1]).eval({t_sigma: sigma}))
    assert np_gk(sigma)(3, 4)!=0
    assert np_gk(sigma)(a[0], a[-1])!=0

def test_gaussian_kernel_with_median():
    assert np.allclose(np_gk(np_l(a))(3, 4), t_gk(t_l(x))(3, 4).eval({x:a}))
    assert np_gk(np_l(a))(3, 4) != 0


def test_HSICS_end_2_end():
    assert np.allclose(
        np_HSIC(a,b,np_gk(np_l(a)),np_gk(np_l(b))),
        t_HSIC(x,y,t_gk(t_l(x)),t_gk(t_l(y))).eval({x:a,y:b})
    )
    assert np.allclose(
        np_HSIC(c, d, np_gk(np_l(c)), np_gk(np_l(d))),
        t_HSIC(x, y, t_gk(t_l(x)), t_gk(t_l(y))).eval({x: c, y: d})
    )

def test_auto_HSIC():
    assert np.allclose(
        t_auto_HSIC(a,b),
        np_auto_HSIC(a,b)
    )
    assert np.allclose(
        t_auto_HSIC(c, d),
        np_auto_HSIC(c, d)
    )

def test_HSIC_purpose():
    assert t_auto_HSIC(a,b) > t_auto_HSIC(c,d)