import numpy as np
import t_HSIC as th
import np_HSIC as nph
import theano.tensor as T

x = T.dvector('x')
y = T.dvector('y')
s_x = T.dscalar('s_x')
s_y = T.dscalar('s_y')

corr = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]],100)
uncorr = np.random.multivariate_normal([0,0],[[1,0],[0,1]],100)

a , b = corr[:,0], corr[:,1]
c , d = uncorr[:,0], uncorr[:,1]

t_x = {x:a}


def test_median():
    assert np.allclose(nph.l(a),th.l(x).eval({x:a}))

def test_gaussian_kernel_same_sigma():
    sigma = nph.l(a)
    th.sigma = T.dscalar("th.sigma")
    assert np.allclose(nph.gk(sigma)(3,4),th.gk(th.sigma)(3,4).eval({th.sigma:sigma}))
    assert np.allclose(nph.gk(sigma)(a[0], a[-1]), th.gk(th.sigma)(a[0], a[-1]).eval({th.sigma: sigma}))
    assert nph.gk(sigma)(3, 4)!=0
    assert nph.gk(sigma)(a[0], a[-1])!=0

def test_gaussian_kernel_with_median():
    assert np.allclose(nph.gk(nph.l(a))(3, 4), th.gk(th.l(x))(3, 4).eval({x:a}))
    assert nph.gk(nph.l(a))(3, 4) != 0


def test_HSICS_end_2_end():
    assert np.allclose(
        nph.HSIC(a,b,nph.gk(nph.l(a)),nph.gk(nph.l(b))),
        th.HSIC(x,y,th.gk(th.l(x)),th.gk(th.l(y))).eval({x:a,y:b})
    )
    assert np.allclose(
        nph.HSIC(c, d, nph.gk(nph.l(c)), nph.gk(nph.l(d))),
        th.HSIC(x, y, th.gk(th.l(x)), th.gk(th.l(y))).eval({x: c, y: d})
    )

def test_auto_HSIC():
    assert np.allclose(
        th.auto_HSIC(a,b),
        nph.auto_HSIC(a,b)
    )
    assert np.allclose(
        th.auto_HSIC(c, d),
        nph.auto_HSIC(c, d)
    )

def test_HSIC_purpose():
    assert th.auto_HSIC(a,b) > th.auto_HSIC(c,d)