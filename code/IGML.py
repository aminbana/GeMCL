import numpy as np
from scipy.special import digamma
from scipy import optimize
from scipy.stats import invgamma
import matplotlib.pyplot as plt

def digammainv(y):
    _em = 0.5772156649015328606065120
    func = lambda x: digamma(x) - y
    if y > -0.125:
        x0 = np.exp(y) + 0.5
        if y < 10:
            value = optimize.newton(func, x0, tol=1e-10)
            return value
    elif y > -3:
        x0 = np.exp(y/2.332) + 0.08661
    else:
        x0 = 1.0 / (-y - _em)

    value, info, ier, mesg = optimize.fsolve(func, x0, xtol=1e-11,
                                             full_output=True)
    if ier != 1:
        raise RuntimeError("digammainv: fsolve failed, y = %r" % y)

    return value[0]

def ML_IG(x, eps=1e-7, n_t=400):
    if np.abs(x).sum() < 0.001:
        print("Data samples are too small")
        return 10, 1

    x = x[x > eps]
    if len(x) < 3:
        print("Data samples are not enough")
        return 4, 2
    n = x.shape[0]
    mu = np.sum(1/x, 0) / n
    nu = np.sum(np.square(1/x - mu), 0) / (n-1)
    
    alpha = mu ** 2 / nu
    C = - np.log(np.sum(1 / x, 0)) - np.sum(np.log(x), 0) / n    
    
    temp = -1
    for i in range(n_t):
        alpha = digammainv(C + np.log(n*alpha))
        if np.min(abs(alpha - temp)) < eps:
            break
        temp = alpha

    beta = n * alpha / np.sum(1 / x, 0)
    return alpha, beta

def Method_of_Moments(samples, offset = None):
    alpha = ((samples).mean())**2 / (samples).var() + 2
    beta = (alpha - 1) * samples.mean()
    return alpha , beta